"""Legacy per-op kernels kept for the TMIX prefill transition.

Before the fused ``kernel.tmix_decode`` / ``kernel.cmix_prefill`` set covered
the whole model, inference ran per-op tilelang kernels bound by dtype
(``fp16``/``bf16`` namespaces over ``_base.build_kernels``). The TMIX prefill
path (``fused_dplr_T`` single-shot recurrence) still uses them. New code
should prefer the fused factories in ``rwkv_tl.kernel``.
"""

from __future__ import annotations

from ._base import Kernels, build_kernels
from .fp16 import (
    fused_dplr,
    fused_dplr_T,
    fused_gates,
    fused_gn_rkrk,
    fused_l2norm_neg_kk_a,
    fused_lerp1_copy,
    fused_lerp6,
    fused_lerp6_copy,
    fused_lerp6_rkv_copy,
    fused_rank_gemv,
    fused_rkv_gemm,
    fused_v_gate,
    fused_w_gate,
    kernels,
)

__all__ = [
    "Kernels",
    "build_kernels",
    "fused_dplr",
    "fused_dplr_T",
    "fused_gates",
    "fused_gn_rkrk",
    "fused_l2norm_neg_kk_a",
    "fused_lerp1_copy",
    "fused_lerp6",
    "fused_lerp6_copy",
    "fused_lerp6_rkv_copy",
    "fused_rank_gemv",
    "fused_rkv_gemm",
    "fused_v_gate",
    "fused_w_gate",
    "kernels",
]
