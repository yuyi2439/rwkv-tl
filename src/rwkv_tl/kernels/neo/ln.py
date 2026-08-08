# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T

_LN_EPS = 1e-5


def ln_macro(LEN, C: int, DTYPE: str, THREADS: int = 256):
    """LayerNorm

    If `x` is useless after layer norm, set `out` the same as `x`.

    `LEN` is the length of data. Look at [tilelang#2916](https://github.com/tile-ai/tilelang/issues/2916)
    """
    assert C % THREADS == 0

    # # Look at https://github.com/tile-ai/tilelang/issues/2916
    # LEN = T.dynamic("LEN")

    @T.macro
    def _impl(
        x: T.Tensor((LEN, C), DTYPE),
        weight: T.Tensor((C,), DTYPE),
        bias: T.Tensor((C,), DTYPE),
        *,
        out: T.Tensor((LEN, C), DTYPE),
    ):
        """LayerNorm
        
        Args:
            out: modified in-place
        """
        with T.Kernel(LEN, threads=THREADS) as n:
            s = T.alloc_fragment((1,), "float32")
            x_frag = T.alloc_fragment((C,), "float32")
            sq_frag = T.alloc_fragment((C,), "float32")

            # copy t
            T.copy(x[n, :], x_frag)

            # sum & mean
            T.reduce_sum(x_frag, s, dim=-1, clear=True)
            mean = s[0] / T.float32(C)  # pyright: ignore[reportCallIssue]

            # (x - mean) ^ 2
            for i in T.Parallel(C):
                x_frag[i] = x_frag[i] - mean
                sq_frag[i] = x_frag[i] * x_frag[i]

            # rstd (reciprocal square root)
            T.reduce_sum(sq_frag, s, dim=-1, clear=True)
            rstd = T.rsqrt(s[0] / T.float32(C) + T.float32(_LN_EPS))  # pyright: ignore[reportCallIssue]

            for i in T.Parallel(C):
                out[n, i] = T.cast(
                    x_frag[i] * rstd * T.cast(weight[i], "float32")
                    + T.cast(bias[i], "float32"),
                    DTYPE,
                )

    return _impl


@tilelang.jit(out_idx=[3])
def ln(C: int, DTYPE: str):
    THREADS = 256
    assert C % THREADS == 0

    LEN = T.dynamic("LEN")

    @T.prim_func
    def _impl(
        x: T.Tensor((LEN, C), DTYPE),
        weight: T.Tensor((C,), DTYPE),
        bias: T.Tensor((C,), DTYPE),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        ln_macro(LEN, C, DTYPE)(x, weight, bias, out=out)

    return _impl
