"""CUDA-Graph acceleration wrapper for any ``RWKV7Model`` implementation.

Merges the old ``graph_decode``/``prefill_graph`` pair into one generic wrapper:
instead of re-implementing the layer loop, ``CUDAGraph`` captures the wrapped
model's OWN ``decode`` and ``prefill`` methods against a fixed-address shadow
``State`` and replays them, copying the caller's ``State`` in/out around each
replay. The wrapped model stays stateless -- any ``State`` works.

Usage::

    model = CUDAGraph(RWKV7MX450(w))        # wrap an existing instance
    # ... or ...
    model = make_rwkv7(device, backend="tuned")   # returns a wrapped class

Capture is lazy (on first call) and per exact prefill length ``T`` up to
``prefill_graph_max_t`` (larger ``T`` runs eager: launch overhead amortizes and
graph memory scales with ``T``). Non-CUDA models, models whose ``prefill``
rebinds ``state["x"]`` instead of updating it in place (CUDA-Graph replay
requires fixed tensor addresses), and any capture failure transparently fall
back to the wrapped model's eager path for the affected op.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
from torch import Tensor

from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

from ._rwkv7_abc import RWKV7Model

__all__ = ["CUDAGraph", "make_graph_cls", "wrap_model"]

_log = logging.getLogger(__name__)


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


class CUDAGraph(RWKV7Model):
    """Wrap an ``RWKV7Model`` instance with CUDA-Graph decode/prefill.

    Args:
        model: Any ``RWKV7Model`` instance (its weights must be on CUDA to
            capture; non-CUDA models are passed through eagerly).
        prefill_graph_max_t: Capture prefill as a graph only for T in
            ``[2, prefill_graph_max_t]``; larger T runs eager.
        warmup: Warmup iterations per captured graph (initialises cuBLAS
            handles / forces lazy allocations so the graph is self-contained).
    """

    def __init__(
        self,
        model: RWKV7Model,
        *,
        prefill_graph_max_t: int = 64,
        warmup: int = 3,
    ) -> None:
        super().__init__(model.w)
        self.model = model
        self.prefill_graph_max_t = prefill_graph_max_t
        self._warmup = warmup

        dev = self.w.device
        self._cuda = torch.cuda.is_available() and dev.type == "cuda"

        # Decode graph state (T=1).
        self._tok: Tensor | None = None
        self._shadow_dec: State | None = None
        self._graph_dec: torch.cuda.CUDAGraph | None = None
        self._logits: Tensor | None = None
        self._dec_attempted = False

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
        if not self._cuda:
            return self.model.decode(token, S)
        if not self._dec_attempted:
            self._dec_attempted = True
            self._capture_decode()
        g = self._graph_dec
        if g is None:
            return self.model.decode(token, S)

        assert self._tok is not None and self._shadow_dec is not None
        self._tok[0] = int(token.item()) if isinstance(token, Tensor) else token
        _copy_state(self._shadow_dec, S)
        g.replay()
        _copy_state(S, self._shadow_dec)
        assert self._logits is not None
        return self._logits.clone(), S

    def prefill(self, tokens: Tensor, S: State) -> State:
        """Batch-fill a token sequence; per-T graph replay when captured."""
        if not self._cuda:
            return self.model.prefill(tokens, S)
        tok = tokens.reshape(-1)
        T = tok.numel()
        if T < 2 or T > self.prefill_graph_max_t:
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

    def _capture_decode(self) -> None:
        """Capture the wrapped model's ``decode`` as a T=1 CUDA graph."""
        if not self._cuda:
            return
        tok = torch.zeros(1, dtype=torch.long, device="cuda")
        shadow = State(self.L, self.C, self.N, device="cuda", dtype=self.w.dtype)
        logits: Tensor | None = None

        def run() -> None:
            nonlocal logits
            out, _ = self.model.decode(tok, shadow)
            logits = out

        g = self._capture_graph(run, shadow, "decode")
        if g is None:
            return
        self._tok = tok
        self._shadow_dec = shadow
        self._graph_dec = g
        self._logits = logits

    def _capture_prefill(self, T: int) -> None:
        """Capture the wrapped model's ``prefill`` for one exact length ``T``."""
        if not self._cuda:
            self._graphs[T] = None
            return
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


def wrap_model(
    model: RWKV7Model,
    *,
    prefill_graph_max_t: int = 64,
    warmup: int = 3,
) -> CUDAGraph:
    """Wrap an existing ``RWKV7Model`` instance with CUDA-Graph acceleration."""
    return CUDAGraph(
        model,
        prefill_graph_max_t=prefill_graph_max_t,
        warmup=warmup,
    )


def make_graph_cls(
    base_cls: type[RWKV7Model],
    *,
    prefill_graph_max_t: int = 64,
    warmup: int = 3,
) -> type[RWKV7Model]:
    """Build a class that constructs ``base_cls(w, **kwargs)`` and wraps it.

    ``make_rwkv7`` returns these so ``model_cls(w, ...)`` transparently yields a
    graph-accelerated model with the same constructor signature.
    """

    def _init(self, w: RWKV7Weight, **kwargs) -> None:
        CUDAGraph.__init__(
            self,
            base_cls(w, **kwargs),
            prefill_graph_max_t=prefill_graph_max_t,
            warmup=warmup,
        )

    return type(f"CUDAGraph{base_cls.__name__}", (CUDAGraph,), {"__init__": _init})
