"""Fused attention reduction kernels: L2-norm gate, group-norm + residual, DPLR
(dtype-parameterized).

Reduction kernels operate over [H, N] (N=HEAD_DIM fixed). Model constants H and
C (= H*N) are baked at COMPILE time: each kernel is compiled per (H) and cached,
so a model's kernels are specialized to its head count (0.1B H=12, 0.4B H=16,
...). Only per-call sizes (token count T) stay dynamic. All arithmetic
accumulates in fp32; results are cast to DTYPE only when written back. The DPLR
RNN state S is always fp32.

``build(DTYPE)`` returns a namespace bound to one element type.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
# NOTE: no `from __future__ import annotations` here -- tilelang's eager builder
# evaluates the annotation expressions, and a stringified annotation would lose
# the closure DTYPE param (NameError).

from types import SimpleNamespace

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from torch import Tensor

from ._common import HEAD_DIM, SERIAL, WARP

N = HEAD_DIM


def build(DTYPE: str) -> SimpleNamespace:
    """Build the dtype-bound attention-reduction kernel namespace.

    Args:
        DTYPE: Element type string, ``"float16"`` or ``"bfloat16"``.

    Returns:
        ``SimpleNamespace`` with ``fused_l2norm_neg_kk_a``,
        ``fused_gn_rkrk``, ``fused_dplr``, ``fused_dplr_T``.
    """

    @tilelang.jit(out_idx=[2, 3])
    def _l2norm_kernel(H: int):
        @T.prim_func
        def _impl(
            kk: T.Tensor((H, N), DTYPE),
            a: T.Tensor((H, N), DTYPE),
            kk_norm_out: T.Tensor((H, N), DTYPE),
            b: T.Tensor((H, N), DTYPE),
        ):
            """Fused L2-normalize(kk) + neg*multiply: B = -(kk/||kk||) * a."""
            with T.Kernel(H, threads=WARP) as h:
                n = T.get_thread_binding(0)
                p_sq = T.alloc_fragment((1,), "float32")
                p_sq[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    idx = n * SERIAL + j
                    v = T.cast(kk[h, idx], "float32")
                    p_sq[0] += v * v
                total_sq = T.warp_reduce_sum(p_sq[0])
                den = T.max(T.sqrt(total_sq), T.float32(1e-12))
                for j in T.serial(SERIAL):
                    idx = n * SERIAL + j
                    kk_norm = T.cast(kk[h, idx], "float32") / den
                    kk_norm_out[h, idx] = T.cast(kk_norm, DTYPE)
                    b[h, idx] = T.cast(-(kk_norm * T.cast(a[h, idx], "float32")), DTYPE)

        return _impl

    @tilelang.jit(out_idx=[7])
    def _gn_rkrk_kernel(H: int):
        @T.prim_func
        def _impl(
            y: T.Tensor((H, N), DTYPE),
            r: T.Tensor((H, N), DTYPE),
            k: T.Tensor((H, N), DTYPE),
            v: T.Tensor((H, N), DTYPE),
            r_k: T.Tensor((H, N), DTYPE),
            ln_xW: T.Tensor((H * N,), DTYPE),
            ln_xB: T.Tensor((H * N,), DTYPE),
            out: T.Tensor((H * N,), DTYPE),
        ):
            """Fused GroupNorm + r*k*r_k residual: y_norm + (sum r*k*r_k) * v."""
            with T.Kernel(H, threads=WARP) as h:
                n = T.get_thread_binding(0)
                p_sum = T.alloc_fragment((1,), "float32")
                p_rkrk = T.alloc_fragment((1,), "float32")
                p_sum[0] = T.float32(0.0)
                p_rkrk[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    idx = n * SERIAL + j
                    p_sum[0] += T.cast(y[h, idx], "float32")
                    p_rkrk[0] += (
                        T.cast(r[h, idx], "float32")
                        * T.cast(k[h, idx], "float32")
                        * T.cast(r_k[h, idx], "float32")
                    )
                total_sum = T.warp_reduce_sum(p_sum[0])
                total_rkrk = T.warp_reduce_sum(p_rkrk[0])
                mean = total_sum / T.float32(N)
                p_var = T.alloc_fragment((1,), "float32")
                p_var[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    idx = n * SERIAL + j
                    diff = T.cast(y[h, idx], "float32") - mean
                    p_var[0] += diff * diff
                total_var = T.warp_reduce_sum(p_var[0])
                rstd = T.float32(1.0) / T.sqrt(
                    total_var / T.float32(N) + T.float32(64e-5)
                )
                for j in T.serial(SERIAL):
                    idx = n * SERIAL + j
                    flat = h * N + idx
                    y_norm = (T.cast(y[h, idx], "float32") - mean) * rstd
                    y_aff = y_norm * T.cast(ln_xW[flat], "float32") + T.cast(
                        ln_xB[flat], "float32"
                    )
                    residual = total_rkrk * T.cast(v[h, idx], "float32")
                    out[flat] = T.cast(y_aff + residual, DTYPE)

        return _impl

    @tilelang.jit(out_idx=[8])
    def _dplr_kernel(H: int):
        @T.prim_func
        def _impl(
            S: T.Tensor((H, N, N), "float32"),
            W: T.Tensor((H, N), DTYPE),
            A: T.Tensor((H, N), DTYPE),
            B: T.Tensor((H, N), DTYPE),
            V: T.Tensor((H, N), DTYPE),
            K: T.Tensor((H, N), DTYPE),
            R: T.Tensor((H, N), DTYPE),
            S_out: T.Tensor((H, N, N), "float32"),
            y_out: T.Tensor((H, N), DTYPE),
        ):
            """Fused DPLR state update: S_new = S*W + (S@A)*B + V (x) K; y = S_new @ R.

            The RNN state S is fp32 (state fp32, IO DTYPE) so it is never
            DTYPE-rounded between steps.
            """
            with T.Kernel(H, N, threads=WARP) as (h, v_n):
                n = T.get_thread_binding(0)
                p_sa = T.alloc_fragment((1,), "float32")
                p_sa[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    a_idx = n * SERIAL + j
                    p_sa[0] += T.cast(S[h, v_n, a_idx], "float32") * T.cast(
                        A[h, a_idx], "float32"
                    )
                sa = T.warp_reduce_sum(p_sa[0])
                v_val = T.cast(V[h, v_n], "float32")
                p_y = T.alloc_fragment((1,), "float32")
                p_y[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    k_idx = n * SERIAL + j
                    s_new = (
                        T.cast(S[h, v_n, k_idx], "float32")
                        * T.cast(W[h, k_idx], "float32")
                        + sa * T.cast(B[h, k_idx], "float32")
                        + v_val * T.cast(K[h, k_idx], "float32")
                    )
                    S_out[h, v_n, k_idx] = s_new
                    p_y[0] += s_new * T.cast(R[h, k_idx], "float32")
                y_val = T.warp_reduce_sum(p_y[0])
                if n == 0:
                    y_out[h, v_n] = T.cast(y_val, DTYPE)

        return _impl

    @tilelang.jit(out_idx=[8])
    def _dplr_T_kernel(H: int):
        T_LEN = T.dynamic("T_LEN")

        @T.prim_func
        def _impl(
            S: T.Tensor((H, N, N), "float32"),
            R: T.Tensor((T_LEN, H, N), DTYPE),
            W: T.Tensor((T_LEN, H, N), DTYPE),
            A: T.Tensor((T_LEN, H, N), DTYPE),
            B: T.Tensor((T_LEN, H, N), DTYPE),
            V: T.Tensor((T_LEN, H, N), DTYPE),
            K: T.Tensor((T_LEN, H, N), DTYPE),
            S_out: T.Tensor((H, N, N), "float32"),
            y_out: T.Tensor((T_LEN, H, N), DTYPE),
        ):
            """Single-shot DPLR over a whole token sequence.

            One launch processes all T tokens: the state recurrence is serialized
            over t inside each (h, v_n) block (parallel over H*N rows), so prefill
            avoids T separate kernel launches.
            """
            with T.Kernel(H, N, threads=WARP) as (h, v_n):
                n = T.get_thread_binding(0)
                for j in T.serial(SERIAL):
                    S_out[h, v_n, n * SERIAL + j] = T.cast(
                        S[h, v_n, n * SERIAL + j], "float32"
                    )
                p_sa = T.alloc_fragment((1,), "float32")
                p_y = T.alloc_fragment((1,), "float32")
                for t in T.serial(T_LEN):
                    p_sa[0] = T.float32(0.0)
                    for j in T.serial(SERIAL):
                        a_idx = n * SERIAL + j
                        p_sa[0] += S_out[h, v_n, a_idx] * T.cast(
                            A[t, h, a_idx], "float32"
                        )
                    sa = T.warp_reduce_sum(p_sa[0])
                    v_val = T.cast(V[t, h, v_n], "float32")
                    p_y[0] = T.float32(0.0)
                    for j in T.serial(SERIAL):
                        k_idx = n * SERIAL + j
                        s_new = (
                            S_out[h, v_n, k_idx] * T.cast(W[t, h, k_idx], "float32")
                            + sa * T.cast(B[t, h, k_idx], "float32")
                            + v_val * T.cast(K[t, h, k_idx], "float32")
                        )
                        S_out[h, v_n, k_idx] = s_new
                        p_y[0] += s_new * T.cast(R[t, h, k_idx], "float32")
                    y_val = T.warp_reduce_sum(p_y[0])
                    if n == 0:
                        y_out[t, h, v_n] = T.cast(y_val, DTYPE)

        return _impl

    def fused_l2norm_neg_kk_a(kk: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        """Fused L2-normalize(kk) + neg*multiply: B = -(kk/||kk||) * a."""
        if kk.device.type != "cuda":
            den = torch.sqrt(torch.sum(kk * kk, dim=1, keepdim=True))
            kk_norm = kk / torch.clamp(den, min=1e-12)
            return kk_norm, -(kk_norm) * a
        return _l2norm_kernel(kk.shape[0])(kk, a)

    def fused_gn_rkrk(
        y: Tensor,
        r: Tensor,
        k: Tensor,
        v: Tensor,
        r_k: Tensor,
        ln_xW: Tensor,
        ln_xB: Tensor,
    ) -> Tensor:
        """Fused GroupNorm + r*k*r_k residual: y_norm + (sum r*k*r_k) * v."""
        if y.device.type != "cuda":
            h, n = y.shape
            y_flat = F.group_norm(y.reshape(1, h * n), h, ln_xW, ln_xB, 64e-5).reshape(
                -1
            )
            rkrk = torch.sum(r * k * r_k, dim=1, keepdim=True)
            return y_flat + (rkrk * v).reshape(-1)
        return _gn_rkrk_kernel(y.shape[0])(y, r, k, v, r_k, ln_xW, ln_xB)

    def fused_dplr(S, R, W, K, V, A, B):
        """Fused DPLR state update: S_new = S*W + (S@A)*B + V (x) K; y = S_new @ R."""
        if S.device.type != "cuda":
            Sf = S.float()
            S_new = (
                torch.einsum("hvk,hk->hvk", Sf, W.float())
                + torch.einsum("hva,ha,hb->hvb", Sf, A.float(), B.float())
                + torch.einsum("hv,hk->hvk", V.float(), K.float())
            )
            y = torch.einsum("hvk,hk->hv", S_new, R.float())
            S.copy_(S_new)
            return y.to(torch_dtype), S
        kernel = _dplr_kernel(S.shape[0])
        y = kernel(S, W, A, B, V, K, R, S)
        return y, S

    def fused_dplr_T(S, R, W, K, V, A, B):
        """Single-shot DPLR state update over a whole token sequence."""
        if S.device.type != "cuda":
            T_len = R.shape[0]
            y = torch.empty(
                T_len, S.shape[0], S.shape[1], dtype=R.dtype, device=S.device
            )
            for t in range(T_len):
                y_t, S = fused_dplr(S, R[t], W[t], K[t], V[t], A[t], B[t])
                y[t] = y_t
            return y, S
        kernel = _dplr_T_kernel(S.shape[0])
        y = kernel(S, R, W, A, B, V, K, S)
        return y, S

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]
    return SimpleNamespace(
        fused_l2norm_neg_kk_a=fused_l2norm_neg_kk_a,
        fused_gn_rkrk=fused_gn_rkrk,
        fused_dplr=fused_dplr,
        fused_dplr_T=fused_dplr_T,
        torch_dtype=torch_dtype,
    )
