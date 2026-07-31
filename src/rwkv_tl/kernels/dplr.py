"""Fused attention reduction kernels: L2-norm gate, group-norm + residual, DPLR.

Reduction kernels operate over [H, N] (N=HEAD_DIM fixed, H dynamic) using a
warp (32 threads) reduction. T.dynamic lets one compilation serve any RWKV7
model (0.1B H=12, 0.4B H=16, ...). All arithmetic accumulates in fp32; results
are cast to bf16 only when written back, trading bit-exactness with PyTorch
eager for fewer casts and better throughput.
"""
from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from torch import Tensor

from ._common import HEAD_DIM, SERIAL, WARP

H = T.dynamic("H")
C = T.dynamic("C")
N = HEAD_DIM


@T.prim_func
def _fused_l2norm_neg_kk_a(
    kk: T.Tensor((H, N), "bfloat16"),
    a: T.Tensor((H, N), "bfloat16"),
    b: T.Tensor((H, N), "bfloat16"),
):
    """Fused L2-normalize(kk) + neg*multiply: B = -(kk/||kk||) * a."""
    for h in T.thread_binding(H, "blockIdx.x"):
        for n in T.thread_binding(WARP, "threadIdx.x"):
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
                b[h, idx] = T.cast(-(kk_norm * T.cast(a[h, idx], "float32")), "bfloat16")


_fused_l2norm_neg_kk_a_kernel = tilelang.compile(_fused_l2norm_neg_kk_a, out_idx=[2])


@T.prim_func
def _fused_gn_rkrk(
    y: T.Tensor((H, N), "bfloat16"),
    r: T.Tensor((H, N), "bfloat16"),
    k: T.Tensor((H, N), "bfloat16"),
    v: T.Tensor((H, N), "bfloat16"),
    r_k: T.Tensor((H, N), "bfloat16"),
    ln_xW: T.Tensor((C,), "bfloat16"),
    ln_xB: T.Tensor((C,), "bfloat16"),
    out: T.Tensor((C,), "bfloat16"),
):
    """Fused GroupNorm + r*k*r_k residual: y_norm + (sum r*k*r_k) * v."""
    for h in T.thread_binding(H, "blockIdx.x"):
        for n in T.thread_binding(WARP, "threadIdx.x"):
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
            rstd = T.float32(1.0) / T.sqrt(total_var / T.float32(N) + T.float32(64e-5))

            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                flat = h * N + idx
                y_norm = (T.cast(y[h, idx], "float32") - mean) * rstd
                y_aff = y_norm * T.cast(ln_xW[flat], "float32") + T.cast(ln_xB[flat], "float32")
                residual = total_rkrk * T.cast(v[h, idx], "float32")
                out[flat] = T.cast(y_aff + residual, "bfloat16")


_fused_gn_rkrk_kernel = tilelang.compile(_fused_gn_rkrk, out_idx=[7])


@T.prim_func
def _fused_dplr(
    S: T.Tensor((H, N, N), "bfloat16"),
    W: T.Tensor((H, N), "bfloat16"),
    A: T.Tensor((H, N), "bfloat16"),
    B: T.Tensor((H, N), "bfloat16"),
    V: T.Tensor((H, N), "bfloat16"),
    K: T.Tensor((H, N), "bfloat16"),
    R: T.Tensor((H, N), "bfloat16"),
    S_out: T.Tensor((H, N, N), "bfloat16"),
    y_out: T.Tensor((H, N), "bfloat16"),
):
    """Fused DPLR state update: S_new = S*W + (S@A)*B + V (x) K; y = S_new @ R."""
    for h in T.thread_binding(H, "blockIdx.y"):
        for v_n in T.thread_binding(N, "blockIdx.x"):
            for n in T.thread_binding(WARP, "threadIdx.x"):
                # sa = sum_a S[h, v_n, a] * A[h, a]
                p_sa = T.alloc_fragment((1,), "float32")
                p_sa[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    a_idx = n * SERIAL + j
                    p_sa[0] += T.cast(S[h, v_n, a_idx], "float32") * T.cast(A[h, a_idx], "float32")
                sa = T.warp_reduce_sum(p_sa[0])

                v_val = T.cast(V[h, v_n], "float32")
                p_y = T.alloc_fragment((1,), "float32")
                p_y[0] = T.float32(0.0)
                for j in T.serial(SERIAL):
                    k_idx = n * SERIAL + j
                    s_new = (
                        T.cast(S[h, v_n, k_idx], "float32") * T.cast(W[h, k_idx], "float32")
                        + sa * T.cast(B[h, k_idx], "float32")
                        + v_val * T.cast(K[h, k_idx], "float32")
                    )
                    S_out[h, v_n, k_idx] = T.cast(s_new, "bfloat16")
                    p_y[0] += s_new * T.cast(R[h, k_idx], "float32")
                y_val = T.warp_reduce_sum(p_y[0])
                if n == 0:
                    y_out[h, v_n] = T.cast(y_val, "bfloat16")


# In-place variant: only y_out (arg 8) is kernel-allocated; S_out (arg 7) is a
# caller-supplied buffer. Passing the SAME tensor as S (arg 0) and S_out (arg 7)
# updates the state in-place, eliminating a state copy. Race-free: each thread
# reads S[h, v_n, {2n, 2n+1}] before writing the same indices.
_fused_dplr_inplace_kernel = tilelang.compile(_fused_dplr, out_idx=[8])


# --------------------------------------------------------------------------- #
#  Python wrappers with CPU fallback
# --------------------------------------------------------------------------- #
def fused_l2norm_neg_kk_a(kk: Tensor, a: Tensor) -> Tensor:
    """Fused L2-normalize(kk) + neg*multiply: B = -(kk/||kk||) * a.

    Args:
        kk: Key-key tensor, [H, N].
        a: Activation gate, [H, N].

    Returns:
        B: -kk_norm * a, [H, N], bf16.
    """
    if kk.device.type != "cuda":
        den = torch.sqrt(torch.sum(kk * kk, dim=1, keepdim=True))
        return -(kk / torch.clamp(den, min=1e-12)) * a
    return _fused_l2norm_neg_kk_a_kernel(kk, a)


def fused_gn_rkrk(
    y: Tensor, r: Tensor, k: Tensor, v: Tensor, r_k: Tensor,
    ln_xW: Tensor, ln_xB: Tensor,
) -> Tensor:
    """Fused GroupNorm + r*k*r_k residual.

    y_norm = group_norm(y, ln_xW, ln_xB, eps=64e-5);
    y_out = y_norm + sum(r*k*r_k, dim=1) * v.

    Args:
        y: DPLR output, [H, N].
        r/k/v: Receptance/key/value, [H, N].
        r_k: r_k residual weight, [H, N].
        ln_xW/ln_xB: GroupNorm affine params, [H*N].

    Returns:
        Output, [H*N], bf16.
    """
    if y.device.type != "cuda":
        h, n = y.shape
        y_flat = F.group_norm(y.reshape(1, h * n), h, ln_xW, ln_xB, 64e-5).reshape(-1)
        rkrk = torch.sum(r * k * r_k, dim=1, keepdim=True)
        return y_flat + (rkrk * v).reshape(-1)
    return _fused_gn_rkrk_kernel(y, r, k, v, r_k, ln_xW, ln_xB)


def fused_dplr(S, R, W, K, V, A, B):
    """Fused DPLR state update: S_new = S*W + (S@A)*B + V (x) K; y = S_new @ R.

    State S is updated in-place; the same tensor is returned as the new state.

    Args:
        S: State, [H, N, N].
        R: Receptance, [H, N].
        W: Decay gate, [H, N].
        K: Key, [H, N].
        V: Value, [H, N].
        A: kk-normalized key, [H, N].
        B: -kk * a, [H, N].

    Returns:
        (y, S): output [H, N] and updated state [H, N, N].
    """
    if S.device.type != "cuda":
        S_new = (
            torch.einsum("hvk,hk->hvk", S, W)
            + torch.einsum("hva,ha,hb->hvb", S, A, B)
            + torch.einsum("hv,hk->hvk", V, K)
        )
        y = torch.einsum("hvk,hk->hv", S_new, R)
        S.copy_(S_new)
        return y, S
    y = _fused_dplr_inplace_kernel(S, W, A, B, V, K, R, S)
    return y, S
