"""RWKV7 model implementations, split by kernel dtype / device strategy.

- ``rwkv7_fp16.RWKV7FP16``: tilelang kernels bound to fp16 (default precision).
- ``rwkv7_bf16.RWKV7BF16``: tilelang kernels bound to bf16 (raw checkpoint
  dtype, no conversion; reference/experimental).
- ``rwkv7_mx450.RWKV7MX450``: fp16 variant tuned for Turing sm_75 — the batch
  prefill GEMMs run in fp32 because Turing fp16 cuBLAS GEMMs are
  pathologically slow for small prefill shapes.
- ``rwkv7_torch.RWKV7Torch``: pure PyTorch reference (kernel-free).

``make_rwkv7`` selects by backend / device / dtype.
"""

from __future__ import annotations

import torch

from rwkv_tl.weight import RWKV7Weight

from ._rwkv7_abc import RWKV7Model
from ._rwkv7_base import LAYER_NORM, RELUSQ, RWKV7Base
from .rwkv7_bf16 import RWKV7BF16
from .rwkv7_fp16 import RWKV7FP16
from .rwkv7_mx450 import RWKV7MX450
from .rwkv7_torch import RWKV7Torch

__all__ = [
    "LAYER_NORM",
    "RELUSQ",
    "RWKV7BF16",
    "RWKV7FP16",
    "RWKV7MX450",
    "RWKV7Base",
    "RWKV7Model",
    "RWKV7Torch",
    "make_rwkv7",
]


def make_rwkv7(
    w: RWKV7Weight,
    *,
    backend: str = "auto",
    is_torch_compile: bool = True,
) -> RWKV7Model:
    """Build a model implementation for a device / dtype.

    Args:
        w: Loaded weights (``rwkv_tl.weight.RWKV7Weight``). For bf16 backends
            load with ``dtype=torch.bfloat16``.
        backend:
            - ``"auto"``: ``RWKV7MX450`` on CUDA sm_75 (Turing), ``RWKV7FP16``
              elsewhere (sm_80+ CUDA / CPU).
            - ``"fp16"``: ``RWKV7FP16``.
            - ``"bf16"``: ``RWKV7BF16`` (weights should be bf16).
            - ``"mx450"``: ``RWKV7MX450`` (fp32 batch GEMMs).
            - ``"torch"``: ``RWKV7Torch`` (pure PyTorch reference).
        is_torch_compile: Compile ``decode`` (see each class).

    Returns:
        A model implementing the ``RWKV7Model`` interface.
    """
    device = w.emb.device
    if backend == "auto":
        if device.type == "cuda" and tuple(torch.cuda.get_device_capability()) < (8, 0):
            return RWKV7MX450(w, is_torch_compile=is_torch_compile)
        return RWKV7FP16(w, is_torch_compile=is_torch_compile)
    if backend == "fp16":
        return RWKV7FP16(w, is_torch_compile=is_torch_compile)
    if backend == "bf16":
        return RWKV7BF16(w, is_torch_compile=is_torch_compile)
    if backend == "mx450":
        return RWKV7MX450(w, is_torch_compile=is_torch_compile)
    if backend == "torch":
        return RWKV7Torch(w, is_torch_compile=is_torch_compile)
    raise ValueError(
        f"unknown backend {backend!r} (expected auto/fp16/bf16/mx450/torch)"
    )
