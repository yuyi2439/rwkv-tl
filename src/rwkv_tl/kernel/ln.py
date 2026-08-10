# pyright: reportInvalidTypeForm=false

import tilelang.language as T

_LN_EPS = 1e-5


def ln_pre_row_macro(LEN, C: int, DTYPE: str):
    """LN_pre of one row: ``x_ln[n, :] = LN_pre(x_row)``.

    Shared by the cmix/tmix prologues so the LN math lives in one place.
    ``x_row`` is a 1-D ``[C]`` view: the whole buffer for decode, or a
    read-only row slice ``x0[n, :]`` for prefill. Writes the normalized row
    to ``x_ln[n, :]``.

    Args:
        LEN: Row count of the ``x_ln`` output buffer.
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
    """

    @T.macro
    def _impl(
        n,
        x_row: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        *,
        x_ln: T.Tensor((LEN, C), DTYPE),
    ):
        s = T.alloc_fragment((1,), "float32")
        x_frag = T.alloc_fragment((C,), "float32")
        sq_frag = T.alloc_fragment((C,), "float32")

        T.copy(x_row, x_frag)

        T.reduce_sum(x_frag, s, dim=-1, clear=True)
        mean = s[0] / T.float32(C)  # pyright: ignore[reportCallIssue]

        # (x - mean) ^ 2
        for i in T.Parallel(C):
            x_frag[i] = x_frag[i] - mean
            sq_frag[i] = x_frag[i] * x_frag[i]

        # rstd (reciprocal square root)
        T.reduce_sum(sq_frag, s, dim=-1, clear=True)
        rstd = T.rsqrt(  # pyright: ignore[reportCallIssue]
            s[0] / T.float32(C) + T.float32(_LN_EPS)  # pyright: ignore[reportCallIssue]
        )

        for i in T.Parallel(C):
            x_ln[n, i] = T.cast(
                x_frag[i] * rstd * T.cast(ln_preW[i], "float32")
                + T.cast(ln_preB[i], "float32"),
                DTYPE,
            )

    return _impl
