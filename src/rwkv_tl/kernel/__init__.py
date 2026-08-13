"""RWKV7 operator library (tilelang).

Weight-bound operator factories: each factory takes the compile-time
hyperparameters (C/DTYPE/H/ranks) **and the weights** at construction and
returns a callable that only needs activations/state at call time::

    ln_pre = ln_kernel(C, DTYPE, ln_preW, ln_preB)
    x_ln = ln_pre(x0)

Both granularities are supported: fine-grained composable operators
(``ln_kernel`` / ``ln_per_row_kernel`` / ``gemv_kernel`` /
``gemv_batch_kernel``) and the coarse fused layer kernels
(``cmix_decode_kernel`` / ``cmix_prefill_kernel`` / ``tmix_decode_kernel`` /
``tmix_prefill_kernel``). The raw ``@tilelang.jit`` factories and shared
macros (``gemv_macro``, ``gemv_main_macro``, ...) stay available for users who
want to build their own fused chains.
"""

from __future__ import annotations

from .cmix import cmix_decode, cmix_decode_kernel, cmix_prefill, cmix_prefill_kernel
from .gemv import (
    gemv_batch_jit,
    gemv_batch_kernel,
    gemv_batch_macro,
    gemv_batch_T_macro,
    gemv_jit,
    gemv_kernel,
    gemv_macro,
    gemv_main_macro,
)
from .ln import (
    ln_jit,
    ln_kernel,
    ln_per_row_jit,
    ln_per_row_kernel,
    ln_per_row_macro,
    ln_prologue_macro,
)
from .tmix import tmix_decode, tmix_decode_kernel, tmix_prefill_kernel

__all__ = [
    "cmix_decode",
    "cmix_decode_kernel",
    "cmix_prefill",
    "cmix_prefill_kernel",
    "gemv_batch_T_macro",
    "gemv_batch_jit",
    "gemv_batch_kernel",
    "gemv_batch_macro",
    "gemv_jit",
    "gemv_kernel",
    "gemv_macro",
    "gemv_main_macro",
    "ln_jit",
    "ln_kernel",
    "ln_per_row_jit",
    "ln_per_row_kernel",
    "ln_per_row_macro",
    "ln_prologue_macro",
    "tmix_decode",
    "tmix_decode_kernel",
    "tmix_prefill_kernel",
]
