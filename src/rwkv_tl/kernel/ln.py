# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T

from ._op import KernelOp

_LN_EPS = 1e-5


def ln_prologue_macro(C: int, DTYPE: str):
    """macro of LayerNorm prologue

    Args:
        C: Channel width.
    """

    @T.macro
    def _impl(
        x: T.Tensor((C,), DTYPE),
    ):
        s = T.alloc_fragment((1,), "float32")
        x_frag = T.alloc_fragment((C,), "float32")
        sq_frag = T.alloc_fragment((C,), "float32")

        T.copy(x, x_frag)

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

        return x_frag, rstd

    return _impl


def ln_per_row_macro(LEN, C: int, DTYPE: str):
    """per row macro of LayerNorm

    Args:
        LEN: Row count of the ``x_ln`` output buffer.
        C: Channel width.
    """

    prologue = ln_prologue_macro(C, DTYPE)

    @T.macro
    def _impl(
        n,
        x_row: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        *,
        x_ln: T.Tensor((LEN, C), DTYPE),
    ):
        x_frag, rstd = prologue(x_row)

        for i in T.Parallel(C):
            x_ln[n, i] = T.cast(
                x_frag[i] * rstd * T.cast(ln_preW[i], "float32")
                + T.cast(ln_preB[i], "float32"),
                DTYPE,
            )

    return _impl


@tilelang.jit(out_idx=[3])
def ln_jit(C: int, DTYPE: str, THREADS: int = 256):
    """Single-token LayerNorm kernel: ``out = LN(x; W, B)``.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block; must divide ``C``.
    """
    assert C % THREADS == 0
    prologue = ln_prologue_macro(C, DTYPE)

    @T.prim_func
    def _impl(
        x: T.Tensor((C,), DTYPE),
        W: T.Tensor((C,), DTYPE),
        B: T.Tensor((C,), DTYPE),
        out: T.Tensor((C,), DTYPE),
    ):
        with T.Kernel(1, threads=THREADS):
            x_frag, rstd = prologue(x)
            for i in T.Parallel(C):
                out[i] = T.cast(
                    x_frag[i] * rstd * T.cast(W[i], "float32")
                    + T.cast(B[i], "float32"),
                    DTYPE,
                )

    return _impl


@tilelang.jit(out_idx=[3])
def ln_per_row_jit(LEN: int, C: int, DTYPE: str, THREADS: int = 256):
    """Per-row LayerNorm kernel over ``[LEN, C]``: ``out[n] = LN(x[n]; W, B)``.

    Args:
        LEN: Row count.
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: Threads per block; must divide ``C``.
    """
    assert C % THREADS == 0
    per_row = ln_per_row_macro(LEN, C, DTYPE)

    @T.prim_func
    def _impl(
        x: T.Tensor((LEN, C), DTYPE),
        W: T.Tensor((C,), DTYPE),
        B: T.Tensor((C,), DTYPE),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        with T.Kernel(LEN, threads=THREADS) as n:
            per_row(n, x[n, :], W, B, x_ln=out)

    return _impl


def ln_kernel(C: int, DTYPE: str, W, B):
    """Bound single-token LayerNorm.

    ``ln = ln_kernel(C, DTYPE, W, B)`` then ``y = ln(x)`` returns
    ``LN(x; W, B)`` with the weights captured at construction.
    """
    return KernelOp(
        ln_jit,
        (C, DTYPE),
        bind={"W": W, "B": B},
        call=("x",),
        name="ln_kernel",
    )


def ln_per_row_kernel(LEN: int, C: int, DTYPE: str, W, B):
    """Bound per-row LayerNorm over ``[LEN, C]`` rows (see ``ln_kernel``)."""
    return KernelOp(
        ln_per_row_jit,
        (LEN, C, DTYPE),
        bind={"W": W, "B": B},
        call=("x",),
        name="ln_per_row_kernel",
    )
