"""RWKV7 model implementations.

- ``rwkv7_tl.RWKV7TL``: fused tilelang kernels (default precision, fp16/bf16).
- ``rwkv7_torch.RWKV7Torch``: pure PyTorch reference (kernel-free).

``make_rwkv7`` selects a model class by backend; on CUDA it returns a class
pre-wrapped in ``CUDAGraph`` (unless ``use_graph=False``), so decode and
per-T prefill run from captured graphs.
"""

from __future__ import annotations

import torch

from ._rwkv7_abc import RWKV7Model
from .cuda_graph import CUDAGraph, make_graph_cls
from .rwkv7_tl import RWKV7TL
from .rwkv7_torch import RWKV7Torch

__all__ = [
    "RWKV7TL",
    "CUDAGraph",
    "RWKV7Model",
    "RWKV7Torch",
    "make_rwkv7",
]


def make_rwkv7(
    device: torch.device,
    *,
    backend: str = "auto",
    use_graph: bool = True,
    device_name: str | None = None,
) -> type[RWKV7Model]:
    """Build a model implementation class for a device.

    Args:
        device: Target device.
        backend:
            - ``"auto"``: ``RWKV7TL`` (fp16 on CUDA ``sm < 80``, else bf16 --
              including ``sm >= 80`` and non-CUDA devices).
            - ``"fp16"``: ``RWKV7TL`` (fp16 weights).
            - ``"bf16"``: ``RWKV7TL`` (bf16 weights; requires sm_80+ tensor
              cores for the fused kernels).
            - ``"tl"``: ``RWKV7TL``.
            - ``"torch"``: ``RWKV7Torch`` (pure PyTorch reference). Like every
                CUDA class it honors ``use_graph``.
            - ``"tuned"``: legacy alias for ``"auto"`` (per-device variants
                were folded into the single ``RWKV7TL``).
        use_graph: Wrap the returned class in a ``CUDAGraph`` so ``decode`` and
            per-T ``prefill`` run from captured CUDA Graphs. Applies to every
            CUDA class; pass ``use_graph=False`` to keep a class truly eager
            (e.g. the torch reference used for correctness gating).
        device_name: Accepted for backward compatibility; no longer selects a
            variant (all CUDA devices use ``RWKV7TL``).

    Returns:
        A class implementing the ``RWKV7Model`` interface.
    """
    cls = _resolve_cls(backend)
    if use_graph and device.type == "cuda":
        return make_graph_cls(cls)
    return cls


def _resolve_cls(backend: str) -> type[RWKV7Model]:
    """Map a backend name to a concrete model class."""
    if backend in ("auto", "fp16", "bf16", "tl", "tuned", "mx450", "rtx3060"):
        return RWKV7TL
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(
        f"unknown backend {backend!r} (expected auto/fp16/bf16/tl/tuned/torch)"
    )
