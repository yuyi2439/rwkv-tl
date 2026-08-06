# pyright: reportInvalidTypeForm=false

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


_BM = 32  # rows per block (MMA M must be a multiple of 16)
_BN = 128  # output-width tile per block
_BK = 32  # K tile
_THREADS = 128
_STAGES = 2


@tilelang.jit
def fused_up_relu2(C: int, DTYPE: str):
    """Fused up-projection + squared-ReLU: h = relu2(x @ kWt).

    2D grid over (hidden columns, token rows); each block computes a
    [BM, BN] hidden tile with the relu^2 epilogue in registers, so the
    [N, 4C] hidden is written to global exactly once (no separate relu kernel).
    Target-agnostic tilelang DSL: it compiles for whatever device the tensors
    live on; the caller picks the target.
    """
    HID = 4 * C
    N = T.dynamic("N")

    @T.prim_func
    def _impl(
        x: T.Tensor((N, C), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        h: T.Tensor((N, HID), DTYPE),
    ):
        with T.Kernel(T.ceildiv(HID, _BN), T.ceildiv(N, _BM), threads=_THREADS) as (
            bu,
            bt,
        ):
            A_sh = T.alloc_shared((_BM, _BK), DTYPE)
            B_sh = T.alloc_shared((_BK, _BN), DTYPE)
            C_frag = T.alloc_fragment((_BM, _BN), "float32")
            T.clear(C_frag)
            for kk in T.Pipelined(T.ceildiv(C, _BK), num_stages=_STAGES):
                T.copy(x[bt * _BM, kk * _BK], A_sh)
                T.copy(kWt[kk * _BK, bu * _BN], B_sh)
                T.gemm(A_sh, B_sh, C_frag)
            for i, j in T.Parallel(_BM, _BN):
                v = C_frag[i, j]
                vv = T.max(v, T.float32(0.0))
                C_frag[i, j] = vv * vv
            T.copy(C_frag, h[bt * _BM, bu * _BN])

    return _impl


@tilelang.jit
def fused_down_add(C: int, DTYPE: str):
    """Fused down-projection + residual add: out = h @ vWt + x0.

    Same 2D structure; the residual is added straight from the accumulate
    fragment in the epilogue (no separate add kernel).
    """
    HID = 4 * C
    N = T.dynamic("N")

    @T.prim_func
    def _impl(
        h: T.Tensor((N, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        x0: T.Tensor((N, C), DTYPE),
        out: T.Tensor((N, C), DTYPE),
    ):
        with T.Kernel(T.ceildiv(C, _BN), T.ceildiv(N, _BM), threads=_THREADS) as (
            bn,
            bt,
        ):
            A_sh = T.alloc_shared((_BM, _BK), DTYPE)
            B_sh = T.alloc_shared((_BK, _BN), DTYPE)
            C_frag = T.alloc_fragment((_BM, _BN), "float32")
            T.clear(C_frag)
            for kk in T.Pipelined(T.ceildiv(HID, _BK), num_stages=_STAGES):
                T.copy(h[bt * _BM, kk * _BK], A_sh)
                T.copy(vWt[kk * _BK, bn * _BN], B_sh)
                T.gemm(A_sh, B_sh, C_frag)
            for i, j in T.Parallel(_BM, _BN):
                out[bt * _BM + i, bn * _BN + j] = T.cast(
                    C_frag[i, j] + T.cast(x0[bt * _BM + i, bn * _BN + j], "float32"),
                    DTYPE,
                )

    return _impl


def fused_cmix(x, kWt, vWt, x0):
    """Fused FFN block: out = x0 + relu2(x @ kWt) @ vWt.

    Pure host glue: pads N to the kernel tile multiple, allocates the hidden
    and output buffers, runs the two tilelang kernels, slices the padding back
    off. Target-agnostic -- device/dtype come from the input tensors; the
    caller decides which target to compile for.
    """
    N, C = x.shape
    dty = x.dtype
    dty_s = "float16" if dty == torch.float16 else "bfloat16"
    pad = (-N) % _BM
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        x0 = F.pad(x0, (0, 0, 0, pad))
        N2 = N + pad
    else:
        N2 = N
    h = torch.empty((N2, 4 * C), dtype=dty, device=x.device)
    out = torch.empty((N2, C), dtype=dty, device=x.device)
    fused_up_relu2(C, dty_s)(x, kWt, h)
    fused_down_add(C, dty_s)(h, vWt, x0, out)
    return out[:N]


@tilelang.jit
def fused_cmix_full_reference(C: int, DTYPE: str):
    """REFERENCE ONLY (correct but slow, ~20x): full one-kernel FFN fusion.

    Keep as a record of why full fusion is not viable: the hidden [BM, 4C]
    forces a 1D grid (one block per row-group serializes the whole hidden
    width), under-parallelizing the GPU, and every block re-reads the full
    weights. Prefer the two 2D-grid kernels above.
    """
    HID = 4 * C
    N = T.dynamic("N")
    BM, BK2, DN_BN, THREADS = 16, 64, 64, 128

    @T.prim_func
    def _impl(
        x: T.Tensor((N, C), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        x0: T.Tensor((N, C), DTYPE),
        out: T.Tensor((N, C), DTYPE),
    ):
        with T.Kernel(T.ceildiv(N, BM), threads=THREADS) as bt:
            x_sh = T.alloc_shared((BM, C), DTYPE)
            A_up = T.alloc_shared((BM, 32), DTYPE)
            B_up = T.alloc_shared((32, BK2), DTYPE)
            up_frag = T.alloc_fragment((BM, BK2), "float32")
            hk_sh = T.alloc_shared((BM, BK2), DTYPE)
            B_dn = T.alloc_shared((BK2, DN_BN), DTYPE)
            dn_frag = T.alloc_fragment((BM, DN_BN), "float32")

            T.copy(x[bt * BM, 0], x_sh)

            for bd in T.serial(T.ceildiv(C, DN_BN)):
                T.clear(dn_frag)
                for kc in T.serial(T.ceildiv(HID, BK2)):
                    T.clear(up_frag)
                    for kk in T.Pipelined(T.ceildiv(C, 32), num_stages=3):
                        T.copy(x_sh[0, kk * 32], A_up)
                        T.copy(kWt[kk * 32, kc * BK2], B_up)
                        T.gemm(A_up, B_up, up_frag)
                    for i, j in T.Parallel(BM, BK2):
                        v = up_frag[i, j]
                        vv = T.max(v, T.float32(0.0))
                        up_frag[i, j] = vv * vv
                    T.copy(up_frag, hk_sh)
                    T.copy(vWt[kc * BK2, bd * DN_BN], B_dn)
                    T.gemm(hk_sh, B_dn, dn_frag)
                for i, j in T.Parallel(BM, DN_BN):
                    out[bt * BM + i, bd * DN_BN + j] = T.cast(
                        dn_frag[i, j]
                        + T.cast(x0[bt * BM + i, bd * DN_BN + j], "float32"),
                        DTYPE,
                    )

    return _impl
