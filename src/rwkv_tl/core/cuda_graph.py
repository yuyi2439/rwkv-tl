"""CUDA-Graph acceleration wrapper for any ``RWKV7Model`` implementation.

Instead of re-implementing the layer loop, ``CUDAGraph`` captures the wrapped
model's OWN ``decode`` and ``prefill`` methods against a fixed-address shadow
``State`` and replays them, copying the caller's ``State`` in/out around each
replay. The wrapped model stays stateless -- any ``State`` works.

Usage::

    model = CUDAGraph(RWKV7TL(w))        # wrap an existing instance
    out = model.generate("...", state=model.new_state())

Capture is lazy (on first call) and per exact prefill length ``T`` up to
``prefill_graph_max_t`` (larger ``T`` runs eager: launch overhead amortizes
and graph memory scales with ``T``). The wrapper is CUDA-only: wrap
conditionally with :func:`try_cuda_graph` and never build ``CUDAGraph``
around a non-CUDA model. Models whose ``prefill`` rebinds ``state["x"]``
instead of updating it in place (CUDA-Graph replay requires fixed tensor
addresses), and any capture failure, transparently fall back to the wrapped
model's eager path for the affected op.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from .model import RWKV7Model
from .state import State

__all__ = ["CUDAGraph", "try_cuda_graph"]


def _copy_state(dst: State, src: State) -> None:
    """Copy all state tensors from ``src`` into ``dst`` in place."""
    for ds, ss in zip(dst.tmix, src.tmix):
        for k in ds:
            ds[k].copy_(ss[k])
    for ds, ss in zip(dst.cmix, src.cmix):
        for k in ds:
            ds[k].copy_(ss[k])


def _state_addrs(state: State) -> list[int]:
    """data_ptr() of every state tensor -- any change means a rebind."""
    addrs: list[int] = []
    for layer in state.tmix + state.cmix:
        for tensor in layer.values():
            addrs.append(tensor.data_ptr())
    return addrs


def try_cuda_graph(model: RWKV7Model, use_graph: bool = True) -> RWKV7Model:
    """Wrap ``model`` in :class:`CUDAGraph` when it runs on CUDA and
    ``use_graph`` is enabled; otherwise return ``model`` unchanged.

    This is the conditional wrapper: decide by the model's actual device (no
    ``torch.cuda.is_available()`` probe) whether to accelerate it with
    CUDA-Graph replay.
    """
    if use_graph and model.w.device.type == "cuda":
        return CUDAGraph(model)
    return model


class CUDAGraph(RWKV7Model):
    """Wrap an ``RWKV7Model`` instance with CUDA-Graph decode/prefill.

    Args:
        model: Any ``RWKV7Model`` instance whose weights are on CUDA. For
            conditional wrapping (CUDA + ``use_graph``), use
            :func:`try_cuda_graph`.
        prefill_graph_max_t: Capture prefill as a graph for T in
            ``[2, prefill_graph_max_t]`` (or any T when ``None``); larger T
            runs eager. The default 1024 covers long prompts (a graph is far
            faster than eager even at T=1024); ``None`` removes the cap.
        warmup: Warmup iterations per captured graph (initialises cuBLAS
            handles / forces lazy allocations so the graph is self-contained).
    """

    def __init__(
        self,
        model: RWKV7Model,
        *,
        prefill_graph_max_t: int | None = 1024,
        warmup: int = 3,
    ) -> None:
        super().__init__(model.w)
        self.model = model
        self.prefill_graph_max_t = prefill_graph_max_t
        self._warmup = warmup

        # Decode graph state (T=1).
        self._tok: Tensor | None = None
        self._shadow_dec: State | None = None
        self._graph_dec: torch.cuda.CUDAGraph | None = None
        self._logits: Tensor | None = None
        self._dec_attempted = False
        self._dec_addrs: list[int] | None = None

        # Per-T prefill graphs.
        self._bufs: dict[int, Tensor] = {}
        self._shadows: dict[int, State] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph | None] = {}

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to the wrapped model (emb, dtype, ...)."""
        model = self.__dict__.get("model")
        if model is not None:
            return getattr(model, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    # ------------------------------------------------------------------ #
    #  RWKV7Model interface
    # ------------------------------------------------------------------ #

    def decode(self, token: Tensor, S: State) -> tuple[Tensor, State]:
        """Advance one token; CUDA-graph replay when captured, eager otherwise."""
        if not self._dec_attempted:
            self._dec_attempted = True
            self._capture_decode(S)
        g = self._graph_dec
        if g is None:
            return self.model.decode(token, S)

        assert self._tok is not None and self._shadow_dec is not None
        self._tok.copy_(token if isinstance(token, Tensor) else torch.tensor(token, dtype=torch.long))
        # Fast path: caller reuses one State (addresses match captured shadow),
        # graph writes S in place -> skip both copies.
        if self._dec_addrs is not None and _state_addrs(S) == self._dec_addrs:
            g.replay()
            assert self._logits is not None
            return self._logits.clone(), S
        # Fallback: copy in -> shadow, replay, copy out -> S (addr mismatch).
        _copy_state(self._shadow_dec, S)
        g.replay()
        _copy_state(S, self._shadow_dec)
        assert self._logits is not None
        return self._logits.clone(), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a token sequence; per-T graph replay when captured."""
        tok = tokens.reshape(-1)
        T = tok.numel()
        if T < 2 or (
            self.prefill_graph_max_t is not None and T > self.prefill_graph_max_t
        ):
            return self.model.prefill(tok, S)
        if T not in self._graphs:
            self._capture_prefill(T)
        g = self._graphs[T]
        if g is None:
            return self.model.prefill(tok, S)

        _copy_state(self._shadows[T], S)
        self._bufs[T].copy_(tok)
        g.replay()
        _copy_state(S, self._shadows[T])
        return S

    # ------------------------------------------------------------------ #
    #  Capture internals
    # ------------------------------------------------------------------ #

    def _capture_decode(self, S: State | None = None) -> None:
        """Capture the wrapped model's ``decode`` as a T=1 CUDA graph.

        The graph captures against ``S`` (or a fresh shadow when ``S`` is None)
        so a caller reusing that exact State can replay with zero copies;
        ``self._dec_addrs`` records the stable addresses.

        Capturing runs ``decode`` a few times on ``shadow`` (warmup + the
        captured step), which advances its contents past the caller's current
        state. We snapshot ``S`` before capture and restore it after so the
        graph's *first* replay starts from exactly the state the caller had --
        subsequent replays keep the shadow and ``S`` in lockstep, so the fast
        path (no copies) stays correct.
        """
        tok = torch.zeros(1, dtype=torch.long, device="cuda")
        shadow = S if S is not None else State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)
        # Snapshot S's contents (addresses unchanged) so we can restore after
        # capture and leave shadow == caller's baseline for the first replay.
        baseline = None
        if S is not None:
            baseline = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)
            _copy_state(baseline, shadow)
        logits: Tensor | None = None

        def run() -> None:
            nonlocal logits
            out, _ = self.model.decode(tok, shadow)
            logits = out

        g = self._capture_graph(run, shadow, "decode")
        if g is None:
            return
        # Roll back the capture's side effect so the first replay starts from
        # the state the caller actually had; fast-path replays then stay in sync.
        if baseline is not None:
            _copy_state(shadow, baseline)
        self._tok = tok
        self._shadow_dec = shadow
        self._graph_dec = g
        self._logits = logits
        self._dec_addrs = _state_addrs(shadow)

    def _capture_prefill(self, T: int) -> None:
        """Capture the wrapped model's ``prefill`` for one exact length ``T``."""
        buf = torch.zeros(T, dtype=torch.long, device="cuda")
        shadow = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)

        def run() -> None:
            self.model.prefill(buf, shadow)

        g = self._capture_graph(run, shadow, f"prefill(T={T})")
        self._graphs[T] = g
        if g is None:
            return
        self._bufs[T] = buf
        self._shadows[T] = shadow

    def _capture_graph(
        self,
        run,
        shadow: State,
        name: str,
    ) -> torch.cuda.CUDAGraph | None:
        """Warm up ``run`` on a side stream and capture it; None if unsafe."""
        try:
            baseline = _state_addrs(shadow)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self._warmup):
                    shadow.reset()
                    run()
            torch.cuda.current_stream().wait_stream(s)
            if _state_addrs(shadow) != baseline:
                warnings.warn(
                    f"CUDA-Graph {name}: the model rebinds state tensors instead of "
                    "updating them in place, so replay would corrupt state; "
                    "falling back to eager.",
                    stacklevel=2,
                )
                return None

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                run()
            torch.cuda.current_stream().synchronize()
            if _state_addrs(shadow) != baseline:
                warnings.warn(
                    f"CUDA-Graph {name}: state tensor addresses changed during "
                    "capture; falling back to eager.",
                    stacklevel=2,
                )
                return None
            return graph
        except Exception as e:  # noqa: BLE001 - capture incompatibility -> eager
            warnings.warn(
                f"CUDA-Graph {name} capture failed ({e}); falling back to eager.",
                stacklevel=2,
            )
            return None
