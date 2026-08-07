# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T


def fused_cmix_main_macro(C: int, DTYPE: str, LEN_block: int):
    """Fused cmix main: x0 + relusq(x @ kWt) @ vWt.

    Args:
        LEN_block: sequence-tile size used to split the input into blocks for
            the kernel. It should be a multiple of 16 and chosen to match the
            typical sequence length for better occupancy.
    """
    assert LEN_block % 16 == 0
    HID = 4 * C
    bytes = T.dtype(DTYPE).bytes  # pyright: ignore[reportCallIssue]

    # TODO: tune these
    _BK = 32
    STAGES = 2
    THREADS = 128
    HID_block = 128

    # shared ≤ 48KB  (note: `^` is XOR, use `**` for power)
    assert (LEN_block * _BK + _BK * HID_block) * STAGES * bytes <= 48 * 2**10

    LEN = T.dynamic("LEN")
    assert HID % HID_block == 0
    assert C % _BK == 0

    @T.macro
    def _impl(
        x: T.Tensor((LEN, C), DTYPE),
        x0: T.Tensor((LEN, C), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        *,
        out: T.Tensor((LEN, C), DTYPE),
    ):
        assert LEN % LEN_block == 0  # pyright: ignore[reportOperatorIssue]
        h = T.alloc_global((LEN, HID), DTYPE)

        # h = relusq(x @ kWt)
        with T.Kernel(T.ceildiv(LEN, LEN_block), HID // HID_block, threads=THREADS) as (
            bx1,
            by1,
        ):
            A_sh = T.alloc_shared((LEN_block, _BK), DTYPE)
            B_sh = T.alloc_shared((_BK, HID_block), DTYPE)
            C_frag = T.alloc_fragment((LEN_block, HID_block), "float32")
            T.clear(C_frag)

            for kk in T.Pipelined(C // _BK, num_stages=STAGES):
                T.copy(x[bx1 * LEN_block, kk * _BK], A_sh)
                T.copy(kWt[kk * _BK, by1 * HID_block], B_sh)
                T.gemm(A_sh, B_sh, C_frag)

            # relusq
            for i, j in T.Parallel(LEN_block, HID_block):
                v = C_frag[i, j]
                vv = T.max(v, T.float32(0.0))  # pyright: ignore[reportCallIssue]
                C_frag[i, j] = vv * vv  # pyright: ignore[reportOperatorIssue]

            # T.copy auto-casts the fp32 fragment to DTYPE on store (T.cast only
            # takes scalars, not a whole buffer).
            T.copy(C_frag, h[bx1 * LEN_block, by1 * HID_block])

        # out = x0 + h @ vWt
        with T.Kernel(LEN // LEN_block, T.ceildiv(C, HID_block), threads=THREADS) as (  # pyright: ignore[reportOperatorIssue]
            bx2,
            by2,
        ):
            A_sh = T.alloc_shared((LEN_block, _BK), DTYPE)
            B_sh = T.alloc_shared((_BK, HID_block), DTYPE)
            C_frag = T.alloc_fragment((LEN_block, HID_block), "float32")
            T.clear(C_frag)

            for kk in T.Pipelined(T.ceildiv(HID, _BK), num_stages=STAGES):
                T.copy(h[bx2 * LEN_block, kk * _BK], A_sh)
                T.copy(vWt[kk * _BK, by2 * HID_block], B_sh)
                T.gemm(A_sh, B_sh, C_frag)

            for i, j in T.Parallel(LEN_block, HID_block):
                out[bx2 * LEN_block + i, by2 * HID_block + j] = T.cast(
                    C_frag[i, j]
                    + T.cast(x0[bx2 * LEN_block + i, by2 * HID_block + j], "float32"),
                    DTYPE,
                )

    return _impl


@tilelang.jit
def fused_cmix():
    pass
