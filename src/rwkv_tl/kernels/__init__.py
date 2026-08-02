"""Tilelang fused kernels for RWKV7.

Kernels are split by function: LERP chains (lerp.py), elementwise gates
(gates.py), and attention reduction kernels -- L2-norm gate, group-norm +
residual, DPLR state update (dplr.py). All kernels use T.dynamic shapes so a
single compilation serves any RWKV7 model.
"""

from __future__ import annotations

from .dplr import (
    fused_dplr,
    fused_gn_rkrk,
    fused_l2norm_neg_kk_a,
)
from .gates import (
    fused_a_kk_k,
    fused_v_gate,
    fused_w_gate,
)
from .gemm import fused_rkv_gemm
from .lerp import (
    fused_lerp1_copy,
    fused_lerp6,
    fused_lerp6_copy,
    fused_lerp6_rkv_copy,
)

__all__ = [
    "fused_a_kk_k",
    "fused_dplr",
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
