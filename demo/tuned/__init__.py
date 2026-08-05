"""Device-tuned RWKV7 model implementations (CUDA-Graph accelerated)."""

from __future__ import annotations

import torch

from .._rwkv7_abc import RWKV7Model

__all__ = ["make_tuned_model"]


def make_tuned_model(device) -> type[RWKV7Model] | None:
    """Select a tuned RWKV7 model class by torch-detected CUDA device.

    Returns a class derived from ``RWKV7Model``, or ``None`` when no tuned
    variant is applicable.
    """
    name = torch.cuda.get_device_name(device).lower().replace(" ", "")
    if "mx450" in name:
        from .rwkv7_mx450 import RWKV7MX450

        return RWKV7MX450
    if "rtx3060" in name:
        from .rwkv7_rtx3060 import RWKV7RTX3060

        return RWKV7RTX3060
    return None
