"""RWKV7 model implementations, split by kernel dtype / device strategy.

- ``rwkv7_fp16.RWKV7FP16``: tilelang kernels bound to fp16 (default precision).
- ``rwkv7_bf16.RWKV7BF16``: tilelang kernels bound to bf16 (raw checkpoint
  dtype, no conversion; reference/experimental).
- ``rwkv7_torch.RWKV7Torch``: pure PyTorch reference (kernel-free).
- ``tuned`` package: per-device CUDA-Graph-accelerated variants —
  ``RWKV7RTX3060`` (Ampere+ sm_80+) and ``RWKV7MX450`` (Turing sm_75, fp32
  batch GEMMs).

``make_rwkv7`` selects by backend / device / dtype.
"""

from __future__ import annotations

import torch

from ._rwkv7_abc import RWKV7Model
from ._rwkv7_base import LAYER_NORM, RELUSQ, RWKV7Base
from .rwkv7_bf16 import RWKV7BF16
from .rwkv7_fp16 import RWKV7FP16
from .rwkv7_torch import RWKV7Torch
from .tuned import make_tuned_model

__all__ = [
    "LAYER_NORM",
    "RELUSQ",
    "RWKV7BF16",
    "RWKV7FP16",
    "RWKV7Base",
    "RWKV7Model",
    "RWKV7Torch",
    "make_rwkv7",
    "make_tuned_model",
]


def make_rwkv7(
    device: torch.device,
    *,
    backend: str = "tuned",
) -> type[RWKV7Model]:
    """Build a model implementation class for a device.

    Args:
        device: Target device.
        backend:
            - ``"auto"``: ``RWKV7FP16`` on CUDA ``sm < 80``, else
                ``RWKV7BF16`` (including ``sm >= 80`` and non-CUDA devices).
            - ``"fp16"``: ``RWKV7FP16``.
            - ``"bf16"``: ``RWKV7BF16`` (weights should be bf16).
            - ``"mx450"``: ``RWKV7MX450`` (Turing sm_75, fp32 batch GEMMs).
            - ``"rtx3060"``: ``RWKV7RTX3060`` (Ampere+ sm_80+, fp16 GEMMs).
            - ``"tuned"``: per-device tuned variant by torch-detected device
                name/capability, falling back to ``"auto"`` for unknown devices.
            - ``"torch"``: ``RWKV7Torch`` (pure PyTorch reference).

    Returns:
        A class implementing the ``RWKV7Model`` interface.
    """
    if backend == "tuned":
        try:
            tuned = make_tuned_model(device)
            if tuned is not None:
                return tuned
        finally:
            backend = "auto"

    if backend == "auto":
        if device.type == "cuda":
            cap = tuple(torch.cuda.get_device_capability(device))
            if cap < (8, 0):
                return RWKV7FP16
        return RWKV7BF16
    if backend == "fp16":
        return RWKV7FP16
    if backend == "bf16":
        return RWKV7BF16
    if backend == "mx450":
        from .tuned.rwkv7_mx450 import RWKV7MX450

        return RWKV7MX450
    if backend == "rtx3060":
        from .tuned.rwkv7_rtx3060 import RWKV7RTX3060

        return RWKV7RTX3060
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(
        f"unknown backend {backend!r} (expected auto/fp16/bf16/mx450/rtx3060/tuned/torch)"
    )
