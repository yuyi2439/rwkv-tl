"""FP16 (``float16``) fused kernels — the default inference precision.

Binds the dtype-parameterized kernels in ``._base`` to ``"float16"``. fp16's
10-bit mantissa exceeds bf16's 7-bit, and fp16 tensor cores work from sm_75
onwards (via cuBLAS), so this is the primary path. The interface is identical
to ``bf16.py``.
"""

from __future__ import annotations

from ._base import Kernels, build_kernels

kernels = build_kernels("float16")

DTYPE = kernels.DTYPE
torch_dtype = kernels.torch_dtype
fused_lerp6 = kernels.fused_lerp6
fused_lerp6_copy = kernels.fused_lerp6_copy
fused_lerp1_copy = kernels.fused_lerp1_copy
fused_lerp6_rkv_copy = kernels.fused_lerp6_rkv_copy
fused_w_gate = kernels.fused_w_gate
fused_v_gate = kernels.fused_v_gate
fused_a_kk_k = kernels.fused_a_kk_k
fused_l2norm_neg_kk_a = kernels.fused_l2norm_neg_kk_a
fused_gn_rkrk = kernels.fused_gn_rkrk
fused_dplr = kernels.fused_dplr
fused_dplr_T = kernels.fused_dplr_T
fused_rkv_gemm = kernels.fused_rkv_gemm

__all__ = [
    "DTYPE",
    "Kernels",
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
    "kernels",
    "torch_dtype",
]
