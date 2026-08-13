"""rwkv_tl: RWKV7 operator library and ready-to-use models.

Build a model directly from a checkpoint path::

    import rwkv_tl

    model = rwkv_tl.rwkv7("model-0.4b.pth")          # backend auto-selected
    text = model.generate("Hello", max_new_tokens=64)
    answer = model.chat([{"role": "user", "content": "Hi!"}])

Low-level core modules (``RWKV7Model`` / ``State`` / ``Tokenizer`` /
``RWKV7Weight`` / ``CUDAGraph``) are NOT re-exported from the top level:
import them from ``rwkv_tl.core`` when needed. ``rwkv7()`` returns the
application-layer ``RWKV7TextModel`` (composition over the token model).
Fused operator factories live in ``rwkv_tl.kernel`` (weight-bound wrappers).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

import torch

from .core.cuda_graph import CUDAGraph
from .rwkv7_tl import RWKV7TL
from .rwkv7_torch import RWKV7Torch
from .sampling import sample_logits
from .text_model import RWKV7TextModel

if TYPE_CHECKING:
    from .core.weight import RWKV7Weight

__all__ = [
    "RWKV7TL",
    "RWKV7TextModel",
    "RWKV7Torch",
    "make_rwkv7",
    "rwkv7",
    "sample_logits",
]

try:
    __version__ = _version("rwkv-tl")
except PackageNotFoundError:  # pragma: no cover - package metadata unavailable
    __version__ = "0.0.0"


def _resolve_cls(backend: str):
    if backend in ("auto",):
        return RWKV7TL if torch.cuda.is_available() else RWKV7Torch
    if backend in ("tl",):
        return RWKV7TL
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(f"unknown backend {backend!r} (expected auto/tl/torch)")


def rwkv7(
    path_or_weight: str | RWKV7Weight,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    backend: str = "auto",
    use_graph: bool = True,
    **kwargs,
) -> RWKV7TextModel:
    """Build an RWKV7 model from a checkpoint path (or ``RWKV7Weight``).

    Args:
        path_or_weight: Checkpoint path or an already-loaded ``RWKV7Weight``.
        device: Target device for checkpoint loading (None = default).
        dtype: Weight precision (``torch.float16`` default; pass
            ``torch.bfloat16`` to keep the raw checkpoint dtype).
        backend: ``"auto"`` selects tilelang on CUDA and pure torch elsewhere;
            ``"tl"`` forces tilelang; ``"torch"`` forces the pure-PyTorch
            reference.
        use_graph: Wrap in CUDA-Graph acceleration when CUDA is available.
        **kwargs: Passed to the model constructor (e.g. ``is_torch_compile``,
            ``cmix_len_block``).
    """
    cls = _resolve_cls(backend)
    model = cls(path_or_weight, device=device, dtype=dtype, **kwargs)
    if use_graph and torch.cuda.is_available() and model.w.device.type == "cuda":
        model = CUDAGraph(model)
    return RWKV7TextModel(model)


def make_rwkv7(
    device: torch.device,
    *,
    backend: str = "auto",
    use_graph: bool = True,
    device_name: str | None = None,
) -> type[RWKV7TextModel]:
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

    def _init(self: RWKV7TextModel, w: RWKV7Weight, **kwargs) -> None:
        model = cls(w, **kwargs)
        if use_graph and device.type == "cuda":
            model = CUDAGraph(model)
        RWKV7TextModel.__init__(self, model)

    return type("RWKV7TextModelFactory", (RWKV7TextModel,), {"__init__": _init})
