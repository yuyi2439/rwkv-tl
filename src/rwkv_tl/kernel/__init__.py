"""Tilelang fused kernels for RWKV7, split by IO dtype.

Kernels are split by function and dtype: the dtype-parameterized
implementations live in ``_base.py``; ``fp16.py`` and ``bf16.py`` bind them to
``float16`` / ``bfloat16`` with identical public interfaces. Everything
accumulates in fp32 internally; only IO tensors use the bound dtype. The
package default below re-exports the fp16 bindings for backward compatibility.

All kernels use T.dynamic shapes so a single compilation serves any RWKV7
model.
"""

from __future__ import annotations

from . import bf16, fp16
from ._base import Kernels, build_kernels
from .fp16 import (
    fused_a_kk_k,
    fused_dplr,
    fused_dplr_T,
    fused_gn_rkrk,
    fused_l2norm_neg_kk_a,
    fused_lerp1_copy,
    fused_lerp6,
    fused_lerp6_copy,
    fused_lerp6_rkv_copy,
    fused_rkv_gemm,
    fused_v_gate,
    fused_w_gate,
)

__all__ = [
    "Kernels",
    "bf16",
    "build_kernels",
    "fp16",
    "fused_a_kk_k",
    "fused_dplr",
    "fused_dplr_T",
    "fused_gn_rkrk",
    "fused_l2norm_neg_kk_a",
    "fused_lerp1_copy",
    "fused_lerp6",
    "fused_lerp6_copy",
    "fused_lerp6_rkv_copy",
    "fused_rkv_gemm",
    "fused_v_gate",
    "fused_w_gate",
]
