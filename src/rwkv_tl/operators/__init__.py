"""PyTorch custom operators wrapping tilelang fused kernels.

Each fused kernel is registered as a `torch.library.custom_op` so that
`torch.compile` (dynamo) treats it as an opaque op instead of graph-breaking
on tilelang internal objects (PrimExprWithOp, Var).

Ops with in-place state mutation (fused_dplr, fused_lerp6_copy,
fused_lerp1_copy) declare `mutates_args` accurately so dynamo preserves the
mutation semantics across the compiled graph.

Fake implementations (register_fake) provide shape inference for dynamo
tracing without invoking the real kernel.
"""

from __future__ import annotations

import torch
from torch import Tensor

_OPS_REGISTERED = False


def _ensure_ops_registered() -> None:
    """Register all custom ops (idempotent via module-level flag)."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return
    _OPS_REGISTERED = True

    # ---- lerp.py ----

    @torch.library.custom_op("rwkv_tl::fused_lerp6_copy", mutates_args=("x_copy",))
    def fused_lerp6_copy_op(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        from ..kernels.lerp import fused_lerp6_copy

        return fused_lerp6_copy(x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy)

    @fused_lerp6_copy_op.register_fake
    def _(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return tuple(torch.empty_like(x) for _ in range(6)) # pyright: ignore[reportReturnType]

    @torch.library.custom_op("rwkv_tl::fused_lerp1_copy", mutates_args=("x_copy",))
    def fused_lerp1_copy_op(
        x: Tensor, prev: Tensor, w: Tensor, x_copy: Tensor
    ) -> Tensor:
        from ..kernels.lerp import fused_lerp1_copy

        return fused_lerp1_copy(x, prev, w, x_copy)

    @fused_lerp1_copy_op.register_fake
    def _(x: Tensor, prev: Tensor, w: Tensor, x_copy: Tensor) -> Tensor:
        return torch.empty_like(x)

    # ---- gates.py ----

    @torch.library.custom_op("rwkv_tl::fused_w_gate", mutates_args=())
    def fused_w_gate_op(x: Tensor, w0: Tensor) -> Tensor:
        from ..kernels.gates import fused_w_gate

        return fused_w_gate(x, w0)

    @fused_w_gate_op.register_fake
    def _(x: Tensor, w0: Tensor) -> Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("rwkv_tl::fused_v_gate", mutates_args=())
    def fused_v_gate_op(v: Tensor, v_first: Tensor, v0: Tensor, v12: Tensor) -> Tensor:
        from ..kernels.gates import fused_v_gate

        return fused_v_gate(v, v_first, v0, v12)

    @fused_v_gate_op.register_fake
    def _(v: Tensor, v_first: Tensor, v0: Tensor, v12: Tensor) -> Tensor:
        return torch.empty_like(v)

    @torch.library.custom_op("rwkv_tl::fused_a_kk_k", mutates_args=())
    def fused_a_kk_k_op(
        a0: Tensor,
        a_x: Tensor,
        k: Tensor,
        k_k: Tensor,
        k_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        from ..kernels.gates import fused_a_kk_k

        return fused_a_kk_k(a0, a_x, k, k_k, k_a)

    @fused_a_kk_k_op.register_fake
    def _(
        a0: Tensor,
        a_x: Tensor,
        k: Tensor,
        k_k: Tensor,
        k_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return torch.empty_like(a0), torch.empty_like(k), torch.empty_like(k)

    # ---- dplr.py ----

    @torch.library.custom_op("rwkv_tl::fused_l2norm_neg_kk_a", mutates_args=())
    def fused_l2norm_neg_kk_a_op(kk: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        from ..kernels.dplr import fused_l2norm_neg_kk_a

        return fused_l2norm_neg_kk_a(kk, a)

    @fused_l2norm_neg_kk_a_op.register_fake
    def _(kk: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        return torch.empty_like(kk), torch.empty_like(kk)

    @torch.library.custom_op("rwkv_tl::fused_gn_rkrk", mutates_args=())
    def fused_gn_rkrk_op(
        y: Tensor,
        r: Tensor,
        k: Tensor,
        v: Tensor,
        r_k: Tensor,
        ln_xW: Tensor,
        ln_xB: Tensor,
    ) -> Tensor:
        from ..kernels.dplr import fused_gn_rkrk

        return fused_gn_rkrk(y, r, k, v, r_k, ln_xW, ln_xB)

    @fused_gn_rkrk_op.register_fake
    def _(
        y: Tensor,
        r: Tensor,
        k: Tensor,
        v: Tensor,
        r_k: Tensor,
        ln_xW: Tensor,
        ln_xB: Tensor,
    ) -> Tensor:
        # Output is flattened [H*N] for decode, or [T, H*N] for batched.
        return torch.empty(
            y.shape[:-2] + (y.shape[-2] * y.shape[-1],), dtype=y.dtype, device=y.device
        )

    @torch.library.custom_op("rwkv_tl::fused_dplr", mutates_args=("S",))
    def fused_dplr_op(
        S: Tensor,
        R: Tensor,
        W: Tensor,
        K: Tensor,
        V: Tensor,
        A: Tensor,
        B: Tensor,
    ) -> Tensor:
        from ..kernels.dplr import fused_dplr

        y, _ = fused_dplr(S, R, W, K, V, A, B)
        return y

    @fused_dplr_op.register_fake
    def _(
        S: Tensor,
        R: Tensor,
        W: Tensor,
        K: Tensor,
        V: Tensor,
        A: Tensor,
        B: Tensor,
    ) -> Tensor:
        return torch.empty_like(R)

    @torch.library.custom_op("rwkv_tl::fused_lerp6_rkv_copy", mutates_args=("x_copy",))
    def fused_lerp6_rkv_copy_op(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
        rWt_stack: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        from ..kernels.lerp import fused_lerp6_rkv_copy

        r, k, v, xv, xw, xa, xg = fused_lerp6_rkv_copy(
            x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy, rWt_stack
        )
        # r/k/v are views of the same stacked tensor; clone to avoid aliasing.
        return r.clone(), k.clone(), v.clone(), xv, xw, xa, xg

    @fused_lerp6_rkv_copy_op.register_fake
    def _(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
        rWt_stack: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # r, k, v have shape [C]; xv, xw, xa, xg have shape [C]
        return tuple(torch.empty_like(x) for _ in range(7)) # pyright: ignore[reportReturnType]

    # ---- gemm.py ----

    @torch.library.custom_op("rwkv_tl::fused_rkv_gemm", mutates_args=())
    def fused_rkv_gemm_op(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
        from ..kernels.gemm import fused_rkv_gemm

        return fused_rkv_gemm(xr, xk, xv, Wb)

    @fused_rkv_gemm_op.register_fake
    def _(xr: Tensor, xk: Tensor, xv: Tensor, Wb: Tensor) -> Tensor:
        # Output: [3, T, C]
        return torch.empty(
            3, xr.shape[0], xr.shape[1], dtype=xr.dtype, device=xr.device
        )


_ensure_ops_registered()
