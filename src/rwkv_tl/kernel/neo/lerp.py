# pyright: reportInvalidTypeForm=false

import tilelang.language as T


def fused_lerp1_macro(LEN, C: int, DTYPE: str):
    # # Look at https://github.com/tile-ai/tilelang/issues/2916
    # LEN = T.dynamic("LEN")

    @T.macro
    def _impl(
        n,
        x: T.Tensor((LEN, C), DTYPE),
        prev: T.Tensor((C,), DTYPE),
        weight: T.Tensor((C,), DTYPE),
        *,
        out: T.Tensor((LEN, C), DTYPE),
    ):
        # read p only
        for i in T.Parallel(C):
            p_val = prev[i] if n == 0 else x[n - 1, i]
            out[n, i] = x[n, i] + weight[i] * (p_val - x[n, i])

    return _impl
