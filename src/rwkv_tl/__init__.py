"""rwkv_tl: RWKV7 operator library and ready-to-use models.

Build a model directly from a checkpoint path::

    import rwkv_tl

    model = rwkv_tl.rwkv7("model-0.4b.pth")          # backend auto-selected
    state = model.tune_state("You are a helpful assistant.")   # state tune
    state.save("persona.pt")
    state = rwkv_tl.State.load("persona.pt")
    out = model.generate("Hello!", state=state)      # text in, text out

Advanced users can drop to the low-level stateless interface
``model.decode(token, state)`` / ``model.prefill(tokens, state)`` /
``model.logits(input, state)`` for raw logits, or compose custom fused
kernels from ``rwkv_tl.kernel`` (weight-bound operator factories).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

import torch

from .cuda_graph import CUDAGraph, make_graph_cls, wrap_model
from .model import RWKV7Model
from .rwkv7_tl import RWKV7TL
from .rwkv7_torch import RWKV7Torch
from .sampling import sample_logits
from .state import State
from .tokenizer import Tokenizer
from .weight import LNWeight, RWKV7Weight

__all__ = [
    "RWKV7TL",
    "CUDAGraph",
    "LNWeight",
    "RWKV7Model",
    "RWKV7Torch",
    "RWKV7Weight",
    "State",
    "Tokenizer",
    "make_rwkv7",
    "rwkv7",
    "sample_logits",
    "wrap_model",
]

try:
    __version__ = _version("rwkv-tl")
except PackageNotFoundError:  # pragma: no cover - package metadata unavailable
    __version__ = "0.0.0"


def _resolve_cls(backend: str):
    if backend in ("auto",):
        return RWKV7TL if torch.cuda.is_available() else RWKV7Torch
    if backend in ("tl", "fp16", "bf16", "tuned", "mx450", "rtx3060"):
        return RWKV7TL
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(
        f"unknown backend {backend!r} (expected auto/tl/fp16/bf16/tuned/torch)"
    )


def rwkv7(
    path_or_weight: str | RWKV7Weight,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    backend: str = "auto",
    use_graph: bool = True,
    **kwargs,
) -> RWKV7Model:
    """Build an RWKV7 model from a checkpoint path (or ``RWKV7Weight``).

    Args:
        path_or_weight: Checkpoint path or an already-loaded ``RWKV7Weight``.
        device: Target device for checkpoint loading (None = default).
        dtype: Weight precision (``torch.float16`` default; pass
            ``torch.bfloat16`` to keep the raw checkpoint dtype).
        backend: ``"auto"`` selects tilelang on CUDA and pure torch elsewhere;
            ``"tl"``/``"fp16"``/``"bf16"`` force tilelang; ``"torch"`` forces
            the pure-PyTorch reference.
        use_graph: Wrap in CUDA-Graph acceleration when CUDA is available.
        **kwargs: Passed to the model constructor (e.g. ``is_torch_compile``,
            ``cmix_len_block``).
    """
    cls = _resolve_cls(backend)
    if backend == "bf16" and dtype is torch.float16:
        # backend="bf16" means keep the raw bf16 checkpoint weights.
        dtype = torch.bfloat16
    model = cls(path_or_weight, device=device, dtype=dtype, **kwargs)
    if use_graph and torch.cuda.is_available() and model.w.device.type == "cuda":
        return CUDAGraph(model)
    return model


def make_rwkv7(
    device: torch.device,
    *,
    backend: str = "auto",
    use_graph: bool = True,
    device_name: str | None = None,
) -> type[RWKV7Model]:
    """Build a model implementation class for a device (class-returning form
    used by benchmark scripts).

    Args:
        device: Target device.
        backend: Same backends as :func:`rwkv7`.
        use_graph: Wrap the returned class so instances are CUDA-Graph
            accelerated (every CUDA class; non-CUDA passes through).
        device_name: Accepted for backward compatibility; no longer selects a
            variant (all CUDA devices use ``RWKV7TL``).
    """
    cls = _resolve_cls(backend)
    if use_graph and device.type == "cuda":
        return make_graph_cls(cls)
    return cls
