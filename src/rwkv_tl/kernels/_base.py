"""Assembles the dtype-parameterized kernel namespaces into one ``Kernels``.

The kernel definitions themselves live in ``gemm.py`` / ``lerp.py`` /
``gates.py`` / ``dplr.py``, each exposing ``build(DTYPE)``. ``build_kernels``
composes them so ``fp16.py`` / ``bf16.py`` can bind one consistent interface.
All kernels accumulate in fp32 internally and only IO tensors use DTYPE, so
the fp32 DPLR RNN state is shared by both variants.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from . import dplr, gates, gemm, lerp

_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class Kernels:
    """Namespace of dtype-bound fused kernels (see ``build_kernels``)."""

    DTYPE: str
    torch_dtype: torch.dtype

    fused_lerp6: Callable[..., tuple[Tensor, ...]]
    fused_lerp6_copy: Callable[..., tuple[Tensor, ...]]
    fused_lerp1_copy: Callable[..., Tensor]
    fused_lerp6_rkv_copy: Callable[..., tuple[Tensor, ...]]
    fused_w_gate: Callable[..., Tensor]
    fused_v_gate: Callable[..., Tensor]
    fused_a_kk_k: Callable[..., tuple[Tensor, Tensor, Tensor]]
    fused_l2norm_neg_kk_a: Callable[..., tuple[Tensor, Tensor]]
    fused_gn_rkrk: Callable[..., Tensor]
    fused_dplr: Callable[..., tuple[Tensor, Tensor]]
    fused_dplr_T: Callable[..., tuple[Tensor, Tensor]]
    fused_rkv_gemm: Callable[..., Tensor]


def build_kernels(DTYPE: str) -> Kernels:
    """Build a dtype-bound kernel namespace.

    Args:
        DTYPE: Element type string, ``"float16"`` or ``"bfloat16"``.

    Returns:
        A ``Kernels`` namespace; every kernel accepts/returns tensors of
        ``DTYPE`` (the DPLR RNN state stays fp32).
    """
    k = Kernels()
    k.DTYPE = DTYPE
    k.torch_dtype = _TORCH_DTYPE[DTYPE]

    g = gemm.build(DTYPE)
    l = lerp.build(DTYPE, g.fused_rkv_gemm)
    ga = gates.build(DTYPE)
    d = dplr.build(DTYPE)

    k.fused_lerp6 = l.fused_lerp6
    k.fused_lerp6_copy = l.fused_lerp6_copy
    k.fused_lerp1_copy = l.fused_lerp1_copy
    k.fused_lerp6_rkv_copy = l.fused_lerp6_rkv_copy
    k.fused_w_gate = ga.fused_w_gate
    k.fused_v_gate = ga.fused_v_gate
    k.fused_a_kk_k = ga.fused_a_kk_k
    k.fused_l2norm_neg_kk_a = d.fused_l2norm_neg_kk_a
    k.fused_gn_rkrk = d.fused_gn_rkrk
    k.fused_dplr = d.fused_dplr
    k.fused_dplr_T = d.fused_dplr_T
    k.fused_rkv_gemm = g.fused_rkv_gemm
    return k
