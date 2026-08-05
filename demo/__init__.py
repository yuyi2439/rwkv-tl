## TODO: 整理这个文件
"""RWKV7 model implementations, split by kernel dtype / device strategy.

- ``rwkv7_fp16.RWKV7FP16``: tilelang kernels bound to fp16 (default precision).
- ``rwkv7_bf16.RWKV7BF16``: tilelang kernels bound to bf16 (raw checkpoint
  dtype, no conversion; reference/experimental).
- ``rwkv7_torch.RWKV7Torch``: pure PyTorch reference (kernel-free).
- ``tuned`` (internal): per-device variants (``RWKV7MX450`` on Turing sm_75).
  Not part of the public API; ``make_rwkv7`` picks the class and callers do not
  need to know which one was selected.

``make_rwkv7`` selects a model class by backend / device; on CUDA it returns a
class pre-wrapped in ``CUDAGraph`` (unless ``use_graph=False``), so decode and
per-T prefill run from captured graphs.
"""

from __future__ import annotations

import torch

from ._rwkv7_abc import RWKV7Model
from ._rwkv7_base import RWKV7Base
from .cuda_graph import CUDAGraph, make_graph_cls
from .rwkv7_bf16 import RWKV7BF16
from .rwkv7_fp16 import RWKV7FP16
from .rwkv7_torch import RWKV7Torch

__all__ = [
    "RWKV7BF16",
    "RWKV7FP16",
    "CUDAGraph",
    "RWKV7Base",
    "RWKV7Model",
    "RWKV7Torch",
    "make_rwkv7",
]

# First CUDA capability with usable bf16 tensor cores; below it the fp16
# kernels are the auto-selected base, at/above it bf16 is the platform default.
_SM80 = (8, 0)


def make_rwkv7(
    device: torch.device,
    *,
    backend: str = "tuned",
    use_graph: bool = True,
    device_name: str | None = None,
) -> type[RWKV7Model]:
    """Build a model implementation class for a device.

    Args:
        device: Target device.
        backend:
            - ``"auto"``: ``RWKV7FP16`` on CUDA ``sm < 80``, else
                ``RWKV7BF16`` (including ``sm >= 80`` and non-CUDA devices).
            - ``"fp16"``: ``RWKV7FP16``.
            - ``"bf16"``: ``RWKV7BF16`` (weights should be bf16).
            - ``"tuned"``: per-device variant selected by CUDA device name;
                unknown CUDA devices fall back to ``RWKV7FP16``, non-CUDA
                devices to ``"auto"``.
            - ``"torch"``: ``RWKV7Torch`` (pure PyTorch reference). Like every
                CUDA class it honors ``use_graph``.
        use_graph: Wrap the returned class in a ``CUDAGraph`` so ``decode`` and
            per-T ``prefill`` run from captured CUDA Graphs. Applies to every
            CUDA class; pass ``use_graph=False`` to keep a class truly eager
            (e.g. the torch reference used for correctness gating).
        device_name: CUDA device name for the ``"tuned"`` backend; when omitted
            it is auto-detected from ``device``. Pass e.g. ``"mx450"`` to force
            a specific tuned variant on any machine.

    Returns:
        A class implementing the ``RWKV7Model`` interface.
    """
    cls = _resolve_cls(device, backend=backend, device_name=device_name)
    if use_graph and device.type == "cuda":
        return make_graph_cls(cls)
    return cls


def _resolve_cls(
    device: torch.device,
    *,
    backend: str,
    device_name: str | None,
) -> type[RWKV7Model]:
    """Map ``(backend, device)`` to a concrete eager model class."""
    if backend == "tuned":
        cls = _select_tuned(device, device_name)
        if cls is not None:
            return cls
        if device.type == "cuda":
            # No per-device variant: fall back to the fp16 base, the default
            # precision the tuned variants build on (also matches the fp16
            # dtype callers load for ``tl-tuned``).
            return RWKV7FP16
        return _resolve_cls(device, backend="auto", device_name=None)
    if backend == "auto":
        if (
            device.type == "cuda"
            and tuple(torch.cuda.get_device_capability(device)) < _SM80
        ):
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
        return RWKV7FP16
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(
        f"unknown backend {backend!r} "
        "(expected auto/fp16/bf16/mx450/rtx3060/tuned/torch)"
    )


def _select_tuned(
    device: torch.device,
    device_name: str | None,
) -> type[RWKV7Model] | None:
    """Pick a per-device tuned class by CUDA device name; ``None`` if none matches.

    An explicit ``device_name`` takes precedence over ``device`` and is honored
    on any device (so the selector can be exercised or forced off-CUDA).
    CUDA introspection failures (uninitialized context, bad device index, ...)
    fall through to ``None`` so the caller can fall back instead of crashing.
    """
    if device_name is None and device.type != "cuda":
        return None
    from .tuned import make_tuned_model

    try:
        name = device_name or torch.cuda.get_device_name(device)
        return make_tuned_model(name)
    except Exception:  # noqa: BLE001 - any CUDA/introspection failure -> no tuned variant
        return None
