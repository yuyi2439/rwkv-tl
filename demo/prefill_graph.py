"""CUDA-Graph batched prefill for one fixed token length (Turing sm_75).

At small T (2..16) MX450 prefill is launch-bound: every layer runs the same op
sequence, so prefill launches a constant ~2175 kernels regardless of T (measured
T=2: GPU 10.7ms but wall ~30ms). A captured graph replays all ~2175 launches as
one graph replay, collapsing the wall time to GPU time.

The model stays stateless: like the decode graph, ``step`` copies the caller's
``State`` into a fixed-address shadow state, replays the graph, and copies back,
so any ``State`` works. The shadow ``state["x"]`` tensors must be updated
in-place by the batch closures (RWKV7MX450 does this).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv_tl.state import State


def _copy_state(dst: State, src: State) -> None:
    """Copy all state tensors from ``src`` into ``dst`` in place."""
    for ds, ss in zip(dst.tmix, src.tmix):
        for k in ds:
            ds[k].copy_(ss[k])
    for ds, ss in zip(dst.cmix, src.cmix):
        for k in ds:
            ds[k].copy_(ss[k])


class PrefillGraph:
    """CUDA-Graph prefill for a fixed token length ``T``.

    Usage::

        pg = PrefillGraph(model, T)   # captures once
        pg.step(token_tensor, state)  # replay, state updated in place
    """

    def __init__(self, model, T: int, warmup: int = 3) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("PrefillGraph requires CUDA")
        if T < 1:
            raise ValueError(f"PrefillGraph needs T>=1, got {T}")
        self.model = model
        self.T = T
        self.token_buf = torch.zeros(T, dtype=torch.long, device="cuda")
        self.state = State(
            model.L,
            model.C,
            model.N,
            device="cuda",
            dtype=model.dtype,
        )
        self._capture(warmup)

    def reset(self) -> None:
        """Zero the shadow state in place (keeps tensor addresses fixed)."""
        self.state.reset()

    def step(self, tokens: torch.Tensor, S: State) -> State:
        """Prefill ``tokens`` into ``S`` via graph replay; returns ``S``."""
        self.token_buf.copy_(tokens.reshape(-1))
        _copy_state(self.state, S)
        self.graph.replay()
        _copy_state(S, self.state)
        return S

    def _run_prefill(self) -> None:
        """One eager prefill pass reading ``token_buf``, updating ``self.state``."""
        X = F.embedding(self.token_buf, self.model.emb)
        v_first: torch.Tensor | None = None
        for (TM, CM), tmix, cmix in zip(
            self.model.layers_batch, self.state.tmix, self.state.cmix
        ):
            X, v_first = TM(X, v_first, tmix)
            X = CM(X, cmix)

    def _capture(self, warmup: int) -> None:
        """Warmup on a side stream then capture the graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.reset()
                self._run_prefill()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            self._run_prefill()
