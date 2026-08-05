"""Eager device-tuned RWKV7 model implementations (internal).

The tuned variants are deliberately eager; CUDA-Graph acceleration is applied
from the outside by ``demo.cuda_graph.CUDAGraph`` (``make_rwkv7(use_graph=True)``
returns pre-wrapped classes).

The selector is internal: callers pass a CUDA device name (auto-detected by
``make_rwkv7``, or supplied manually to force a variant) and only ``make_rwkv7``
consumes the result.
"""

from __future__ import annotations

from .._rwkv7_abc import RWKV7Model

__all__ = ["make_tuned_model"]


def make_tuned_model(device_name: str) -> type[RWKV7Model] | None:
    """Select a tuned RWKV7 model class by CUDA device name.

    Args:
        device_name: CUDA device name (e.g. ``torch.cuda.get_device_name()``),
            or a substring used to force a specific variant.

    Returns:
        A class derived from ``RWKV7Model``, or ``None`` when no tuned
        variant is applicable.
    """
    name = device_name.lower().replace(" ", "")
    if "mx450" in name:
        from .rwkv7_mx450 import RWKV7MX450

        return RWKV7MX450
    if "rtx3060" in name:
        # base fp16 is now graph-capturable (copy_ in base); RTX3060 needs no
        # dedicated tuning any more.
        from ..rwkv7_fp16 import RWKV7FP16

        return RWKV7FP16
    return None
