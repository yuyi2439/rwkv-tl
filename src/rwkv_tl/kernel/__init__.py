"""Fused RWKV7 kernels (dtype-parameterized factories).

The modern kernel set replaces the old per-op ``fp16``/``bf16`` bindings with
dtype-parameterized factories: call ``cmix_decode(C, DTYPE)`` /
``cmix_prefill(C, DTYPE, LEN_block)`` / ``tmix_decode(C, DTYPE, H, Rv, Rw, Ra,
Rg)`` and call the returned kernel directly. ``gemv_macro`` /
``gemv_main_macro`` / ``gemv_batch_macro`` / ``ln_per_row_macro`` /
``ln_prologue_macro`` are the shared building blocks.

Prefill TMIX still uses the legacy per-op kernels during the transition; they
live under ``rwkv_tl.kernel.old`` (``build_kernels(DTYPE)`` returns the old
``Kernels`` namespace).
"""

from __future__ import annotations

from .cmix import cmix_decode, cmix_prefill
from .gemv import gemv_batch_macro, gemv_macro, gemv_main_macro
from .ln import ln_per_row_macro, ln_prologue_macro
from .tmix import tmix_decode

__all__ = [
    "cmix_decode",
    "cmix_prefill",
    "gemv_batch_macro",
    "gemv_macro",
    "gemv_main_macro",
    "ln_per_row_macro",
    "ln_prologue_macro",
    "tmix_decode",
]
