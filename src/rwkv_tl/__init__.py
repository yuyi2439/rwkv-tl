"""rwkv_tl: RWKV7 operator library and ready-to-use models.

The caller owns the steps; nothing is auto-detected. Load a weight first,
then build the token-level model, then optionally wrap it::

    import rwkv_tl
    from rwkv_tl.core import RWKV7Weight

    w = RWKV7Weight("model-0.4b.pth", device="cuda")
    model = rwkv_tl.rwkv7_model(w, backend="tl")   # token-level RWKV7Model
    text = rwkv_tl.RWKV7TextModel(model)           # text-level wrapper
    # one-call convenience (same steps, wrapped):
    text = rwkv_tl.rwkv7(w, backend="tl")

``rwkv7_model()`` maps a loaded ``RWKV7Weight`` to the backend
``RWKV7Model`` (optionally CUDA-Graph wrapped); ``rwkv7()`` is the one-call
form that returns the ``RWKV7TextModel``. Low-level core modules
(``RWKV7Model`` / ``State`` / ``Tokenizer`` / ``RWKV7Weight`` /
``CUDAGraph``) are NOT re-exported from the top level: import them from
``rwkv_tl.core`` when needed. Fused operator factories live in
``rwkv_tl.kernel`` (weight-bound wrappers).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

from .core.cuda_graph import try_cuda_graph
from .rwkv7_tl import RWKV7TL
from .rwkv7_torch import RWKV7Torch
from .sampling import sample_logits
from .text_model import RWKV7TextModel

if TYPE_CHECKING:
    from .core.model import RWKV7Model
    from .core.weight import RWKV7Weight

__all__ = [
    "RWKV7TL",
    "RWKV7TextModel",
    "RWKV7Torch",
    "rwkv7",
    "rwkv7_model",
    "sample_logits",
]

try:
    __version__ = _version("rwkv-tl")
except PackageNotFoundError:  # pragma: no cover - package metadata unavailable
    __version__ = "0.0.0"


def _resolve_cls(backend: str):
    if backend in ("tl",):
        return RWKV7TL
    if backend == "torch":
        return RWKV7Torch
    raise ValueError(f"unknown backend {backend!r} (expected tl/torch)")


def rwkv7_model(
    w: RWKV7Weight,
    *,
    backend: str,
    use_graph: bool = True,
    **kwargs,
) -> RWKV7Model:
    """Build a token-level ``RWKV7Model`` from an already-loaded weight.

    The caller creates the ``RWKV7Weight`` first (device/dtype are fixed at
    load); this function only selects the backend implementation and
    optionally CUDA-Graph-wraps it. Wrap the result in ``RWKV7TextModel``
    for the text-level API, or use :func:`rwkv7` for the one-call form.

    Args:
        w: Already-loaded ``RWKV7Weight``.
        backend: Required backend name: ``"tl"`` (tilelang, CUDA) or
            ``"torch"`` (pure-PyTorch, CPU-capable). There is no
            auto-detection.
        use_graph: Wrap in CUDA-Graph acceleration when the model is on CUDA.
        **kwargs: Passed to the model constructor (e.g. ``is_torch_compile``,
            ``cmix_len_block``).
    """
    model = _resolve_cls(backend)(w, **kwargs)
    return try_cuda_graph(model, use_graph)


def rwkv7(
    w: RWKV7Weight,
    *,
    backend: str,
    use_graph: bool = True,
    **kwargs,
) -> RWKV7TextModel:
    """Build a ``RWKV7TextModel`` from an already-loaded weight.

    One-call convenience for ``RWKV7TextModel(rwkv7_model(w, ...))``; the
    caller still creates the ``RWKV7Weight`` manually. For the decoupled
    two-step path (token model, then wrap), use :func:`rwkv7_model`.

    Args:
        w: Already-loaded ``RWKV7Weight``.
        backend: Required backend name: ``"tl"`` (tilelang, CUDA) or
            ``"torch"`` (pure-PyTorch, CPU-capable). There is no
            auto-detection.
        use_graph: Wrap in CUDA-Graph acceleration when the model is on CUDA.
        **kwargs: Passed to the model constructor (e.g. ``is_torch_compile``,
            ``cmix_len_block``).
    """
    return RWKV7TextModel(
        rwkv7_model(w, backend=backend, use_graph=use_graph, **kwargs)
    )
