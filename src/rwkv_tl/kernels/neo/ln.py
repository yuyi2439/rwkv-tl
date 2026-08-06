# pyright: reportInvalidTypeForm=false


import tilelang
import tilelang.language as T

_LN_EPS = 1e-5


def ln_macro(C: int, DTYPE: str):
    """LayerNorm

    If `x` is useless after layer norm, set `out` the same as `x`.

    `N` is the count of data.

    `n` takes each integer in [0, N)
    """
    N = T.dynamic("N")

    @T.macro
    def _impl(
        x: T.Tensor((N, C), DTYPE),
        out: T.Tensor((N, C), DTYPE),
        weight: T.Tensor((C,), DTYPE),
        bias: T.Tensor((C,), DTYPE),
        n,
    ):
        s = T.alloc_fragment((1,), "float32")
        x_frag = T.alloc_fragment((C,), "float32")
        sq_frag = T.alloc_fragment((C,), "float32")

        # copy t
        T.copy(x[n, :], x_frag)

        # sum & mean
        T.reduce_sum(x_frag, s, dim=-1, clear=True)
        mean = s[0] / T.float32(C)

        # x - mean
        for i in T.Parallel(C):
            x_frag[i] = x_frag[i] - mean

        # rstd (reciprocal square root)
        for i in T.Parallel(C):
            sq_frag[i] = x_frag[i] * x_frag[i]
        T.reduce_sum(sq_frag, s, dim=-1, clear=True)
        rstd = T.rsqrt(s[0] / T.float32(C) + T.float32(_LN_EPS))

        for i in T.Parallel(C):
            out[n, i] = T.cast(
                x_frag[i] * rstd * T.cast(weight[i], "float32")
                + T.cast(bias[i], "float32"),
                DTYPE,
            )

    return _impl


@tilelang.jit
def ln(C: int, DTYPE: str):
    THREADS = 256
    assert C % THREADS == 0

    N = T.dynamic("N")
    ln_impl = ln_macro(C, DTYPE)

    @T.prim_func
    def _impl(
        x: T.Tensor((N, C), DTYPE),
        out: T.Tensor((N, C), DTYPE),
        w: T.Tensor((C,), DTYPE),
        b: T.Tensor((C,), DTYPE),
    ):
        with T.Kernel(N, threads=THREADS) as n:
            ln_impl(x, out, w, b, n)

    return _impl
