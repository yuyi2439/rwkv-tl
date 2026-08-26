"""CUDA-Graph acceleration wrapper for any ``RWKV7Model`` implementation.

Instead of re-implementing the layer loop, ``CUDAGraph`` captures the wrapped
model's OWN ``decode`` and ``prefill`` methods against fixed-address buffers
and replays them. Replay writes to fixed tensor addresses, so the wrapper is
STATEFUL BY DESIGN: it owns exactly one ``State`` (``self.state``), and only
calls that pass it get graph replay. Any other ``State`` runs the wrapped
model eagerly (correct, unaccelerated) -- there is deliberately no copy-in/out
bridge, because routing foreign states through one graph is exactly the
aliasing hazard this single-owner design removes. Backends stay stateless and
shareable; the graph wrapper is the one stateful singleton on top.

Usage::

    model = try_cuda_graph(RWKV7TL(w))   # conditional wrap
    S = model.state                      # the wrapper's own state
    logits, S = model.decode(tok, S)     # replay: zero state copies

Capture is lazy (on first call) and per exact prefill length ``T`` up to
``prefill_graph_max_t`` (larger ``T`` runs eager: launch overhead amortizes
and graph memory scales with ``T``). The wrapper is CUDA-only: wrap
conditionally with :func:`try_cuda_graph` and never build ``CUDAGraph``
around a non-CUDA model. Models whose ``prefill`` rebinds ``state["x"]``
instead of updating it in place (replay requires fixed tensor addresses), and
any capture failure, transparently fall back to the wrapped model's eager
path for the affected op.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from .model import RWKV7Model
from .state import State

__all__ = ["CUDAGraph", "try_cuda_graph"]


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

    The wrapper owns one internal ``State`` (``self.state``). Passing it to
    ``decode``/``prefill`` replays the graph with zero state copies; passing
    any other ``State`` runs the wrapped model eagerly.

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

        # The wrapper's OWN state: the only State that gets graph replay.
        self.state = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)

        # Decode graph state (T=1).
        self._tok: Tensor | None = None
        self._graph_dec: torch.cuda.CUDAGraph | None = None
        self._logits: Tensor | None = None
        self._dec_attempted = False

        # Per-T prefill graphs.
        self._bufs: dict[int, Tensor] = {}
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
        """Advance one token.

        ``S is self.state`` -> graph replay (zero state copies) once captured;
        any other ``State`` -> the wrapped model's eager decode.
        """
        if S is not self.state:
            return self.model.decode(token, S)
        if not self._dec_attempted:
            self._dec_attempted = True
            self._capture_decode()
        g = self._graph_dec
        if g is None:
            return self.model.decode(token, S)

        assert self._tok is not None
        self._tok.copy_(
            token
            if isinstance(token, Tensor)
            else torch.tensor(token, dtype=torch.long)
        )
        g.replay()
        assert self._logits is not None
        return self._logits.clone(), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a token sequence.

        ``S is self.state`` -> per-T graph replay once captured; any other
        ``State`` (or T outside the graph range) -> eager prefill.
        """
        tok = tokens.reshape(-1)
        T = tok.numel()
        if (
            S is not self.state
            or T < 2
            or (self.prefill_graph_max_t is not None and T > self.prefill_graph_max_t)
        ):
            return self.model.prefill(tok, S)
        if T not in self._graphs:
            self._capture_prefill(T)
        g = self._graphs[T]
        if g is None:
            return self.model.prefill(tok, S)

        self._bufs[T].copy_(tok)
        g.replay()
        return S

    # ------------------------------------------------------------------ #
    #  Capture internals
    # ------------------------------------------------------------------ #

    def _capture_decode(self) -> None:
        """Capture the wrapped model's ``decode`` as a T=1 CUDA graph.

        Warmup + capture advance ``self.state``; snapshot it before and
        restore after so the first replay starts from the caller's actual
        state.
        """
        tok = torch.zeros(1, dtype=torch.long, device="cuda")
        baseline = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)
        for i in range(self.L):
            for k in self.state.tmix[i]:
                baseline.tmix[i][k].copy_(self.state.tmix[i][k])
            for k in self.state.cmix[i]:
                baseline.cmix[i][k].copy_(self.state.cmix[i][k])
        logits: Tensor | None = None

        def run() -> None:
            nonlocal logits
            out, _ = self.model.decode(tok, self.state)
            logits = out

        g = self._capture_graph(run, self.state, "decode")
        if g is None:
            return
        # Roll back the capture's side effect so the first replay starts from
        # the state the caller actually had.
        for i in range(self.L):
            for k in self.state.tmix[i]:
                self.state.tmix[i][k].copy_(baseline.tmix[i][k])
            for k in self.state.cmix[i]:
                self.state.cmix[i][k].copy_(baseline.cmix[i][k])
        self._tok = tok
        self._graph_dec = g
        self._logits = logits

    def _capture_prefill(self, T: int) -> None:
        """Capture the wrapped model's ``prefill`` for one exact length ``T``.

        Same snapshot/restore as :meth:`_capture_decode`: warmup + capture
        must not destroy the caller's current state.
        """
        buf = torch.zeros(T, dtype=torch.long, device="cuda")
        baseline = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)
        for i in range(self.L):
            for k in self.state.tmix[i]:
                baseline.tmix[i][k].copy_(self.state.tmix[i][k])
            for k in self.state.cmix[i]:
                baseline.cmix[i][k].copy_(self.state.cmix[i][k])

        def run() -> None:
            self.model.prefill(buf, self.state)

        g = self._capture_graph(run, self.state, f"prefill(T={T})")
        self._graphs[T] = g
        if g is None:
            return
        for i in range(self.L):
            for k in self.state.tmix[i]:
                self.state.tmix[i][k].copy_(baseline.tmix[i][k])
            for k in self.state.cmix[i]:
                self.state.cmix[i][k].copy_(baseline.cmix[i][k])
        self._bufs[T] = buf

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
