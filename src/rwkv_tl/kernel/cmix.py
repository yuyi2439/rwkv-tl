# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T

from ._common import WARP
from .gemv import gemv_macro
from .ln import _LN_EPS, ln_pre_row_macro


def _relusq(v, _idx):
    """fp32 relu then square (cmix hidden activation)."""
    vv = T.max(v, T.float32(0.0))  # pyright: ignore[reportCallIssue]
    return vv * vv  # pyright: ignore[reportOperatorIssue]


def _make_add_residual(x0):
    """Down-pass epilogue factory: ``out = acc + x0``.

    ``gemv_macro``'s epilogue takes ``(acc, idx)`` only; the residual tensor is
    captured here by closure so the GEMV kernel stays residual-agnostic.
    """

    def _add_residual(v, idx):
        return v + T.cast(x0[idx], "float32")

    return _add_residual


##### decode


def cmix_decode_prologue_macro(C: int, DTYPE: str, THREADS: int = 256):
    """Decode cmix prologue: ``x = LN_pre(x0) + x_k * (prev_x - LN_pre(x0))``.

    Fuses the whole prologue into ONE kernel (decode is the hot path; every
    extra launch costs wall time): LN_pre leaves its normalized row in
    registers, the token-shift lerp consumes it directly (no global ``x_ln``
    round-trip), and ``prev_x`` is overwritten in place -- each thread reads
    and writes the same ``prev_x[i]`` element, so there is no cross-thread
    race. ``x_frag`` is indexed ``i * THREADS + tid``, so the LN write and the
    lerp read are the same thread's own element.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block; must divide ``C``.
    """

    # TODO: tune the THREADS parameter
    assert C % THREADS == 0

    # LEN = T.dynamic("LEN")

    @T.macro
    def _impl(
        x0: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        out: T.Tensor((C,), DTYPE),
    ):
        """1 kernel.

        Args:
            x0: raw token ``[C]``.
            ln_preW: LN_pre weight ``[C]``.
            ln_preB: LN_pre bias ``[C]``.
            x_k: token-shift mix weight ``[C]``.
            prev_x: token-shift state ``[C]``; updated in place to ``LN_pre(x0)``.
            out: output ``[C]`` = ``LN_pre(x0) + x_k * (prev_x - LN_pre(x0))``.
        """
        # LN math inlined (keeps the normalized row in registers for the lerp).
        with T.Kernel(1, threads=THREADS):
            s = T.alloc_fragment((1,), "float32")
            x_frag = T.alloc_fragment((C,), "float32")
            sq_frag = T.alloc_fragment((C,), "float32")

            T.copy(x0, x_frag)

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
                ln_val = x_frag[i] * rstd * T.cast(ln_preW[i], "float32") + T.cast(
                    ln_preB[i], "float32"
                )
                # lerp against the OLD prev_x, then store the new LN row to it.
                out[i] = T.cast(
                    ln_val
                    + T.cast(x_k[i], "float32")
                    * (T.cast(prev_x[i], "float32") - ln_val),
                    DTYPE,
                )
                prev_x[i] = T.cast(ln_val, DTYPE)

    return _impl


def cmix_decode_main_macro(C: int, DTYPE: str, THREADS: int = WARP):
    """Fused single-token cmix main: ``out = x0 + relusq(x @ kWt) @ vWt``.

    Decode version of ``cmix_prefill_main_macro``: x is one token ``[C]``
    instead of ``[LEN, C]``, so the two GEMMs collapse to two GEMVs. Both are
    ``gemv_macro`` calls with the epilogue fused in: ``relusq`` on the up pass
    and the ``x0`` residual add on the down pass.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block for both GEMV kernels; must divide ``C``
            and ``4*C``.
    """
    HID = 4 * C
    assert C % THREADS == 0
    assert HID % THREADS == 0

    @T.macro
    def _impl(
        x: T.Tensor((C,), DTYPE),
        x0: T.Tensor((C,), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        *,
        out: T.Tensor((C,), DTYPE),
    ):
        """2 kernels."""
        h = T.alloc_global((HID,), DTYPE)

        # h = relusq(x @ kWt)
        up = gemv_macro(HID, C, DTYPE, THREADS, epilogue=_relusq)
        with T.Kernel(HID // THREADS, threads=THREADS) as bx:
            up(bx, x, kWt, out=h)

        # out = x0 + h @ vWt
        down = gemv_macro(C, HID, DTYPE, THREADS, epilogue=_make_add_residual(x0))
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            down(bx, h, vWt, out=out)

    return _impl


@tilelang.jit(out_idx=[7])
def cmix_decode(C: int, DTYPE: str):
    """Fused single-token cmix: LN_pre + token-shift + relusq(x @ kWt) @ vWt.

    ``cmix_prefill`` with LEN pinned to 1. The prologue runs on the single
    ``[C]`` token in one block (so the LN->lerp read-after-write is race-free
    without a separate kernel), and the two GEMMs collapse to two
    ``gemv_macro`` GEMVs.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block for the elementwise prologue and both GEMV
            kernels; must divide ``C`` and ``4*C``.
    """
    HID = 4 * C
    prologue = cmix_decode_prologue_macro(C, DTYPE)
    main = cmix_decode_main_macro(C, DTYPE)

    @T.prim_func
    def _impl(
        x0: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        out: T.Tensor((C,), DTYPE),
    ):
        """3 kernels."""
        x = T.alloc_global((C,), DTYPE)
        prologue(x0, ln_preW, ln_preB, x_k, prev_x=prev_x, out=x)

        # main (kernels 2-3): out = x0 + relusq(x @ kWt) @ vWt
        main(x, x0, kWt, vWt, out=out)

    return _impl


##### prefill


def cmix_prefill_prologue_macro(
    LEN, C: int, DTYPE: str, THREADS: int = 256, recompute: bool = True
):
    """Prefill cmix prologue: ``x = LN_pre(x0) + x_k * (prev - LN_pre(x0))``.

    Two strategies, selected by ``recompute`` (default True = measured faster;
    see below).

    - ``recompute=True`` (2 kernels): the lerp recomputes ``LN_pre(x0[n-1])``
      locally inside each block (only reads the immutable ``x0``, no cross-block
      dependency, so LN+lerp fuse into one kernel) and only the last block's LN
      row is kept for the ``prev_x`` copy. Saves one launch at the cost of
      doubling the LN work; measured 14-32% faster on MX450 at T=32..256, so
      it is the default.
    - ``recompute=False`` (3 kernels): LN over all tokens in kernel 1, then the
      token-shift lerp in kernel 2 reading ``x_ln[n-1]`` (written by block
      ``n-1`` in kernel 1 -- safe because kernel 1 completed before kernel 2
      launched), then the ``prev_x`` copy. Every block runs one LN.

    ``prev_x`` is updated in place to the LN output of the last token.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block; must divide ``C``.
        recompute: if True, fuse LN+lerp into one kernel by recomputing each
            block's shift source instead of reading ``x_ln[n-1]``.
    """

    # TODO: tune the THREADS parameter
    assert C % THREADS == 0

    ln = ln_pre_row_macro(LEN, C, DTYPE)

    @T.macro
    def _impl(
        x0: T.Tensor((LEN, C), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        """Args:
        x0: raw tokens ``[LEN, C]``.
        ln_preW: LN_pre weight ``[C]``.
        ln_preB: LN_pre bias ``[C]``.
        x_k: token-shift mix weight ``[C]``.
        prev_x: token-shift state ``[C]``; updated in place to ``LN_pre``
            of the last token.
        out: output ``[LEN, C]`` =
            ``LN_pre(x0) + x_k * (prev - LN_pre(x0))`` where ``prev`` is
            ``prev_x`` for the first token else the previous token's
            ``LN_pre`` output.
        """
        if recompute:
            x_last = T.alloc_global((1, C), DTYPE)

            # kernel 1: LN + lerp fused; each block recomputes its own shift
            # source from x0 (immutable), so no cross-block dependency.
            with T.Kernel(LEN, threads=THREADS) as n:
                s = T.alloc_fragment((1,), "float32")
                x_frag = T.alloc_fragment((C,), "float32")
                sq_frag = T.alloc_fragment((C,), "float32")

                # LN_pre(x0[n])
                T.copy(x0[n, :], x_frag)
                T.reduce_sum(x_frag, s, dim=-1, clear=True)
                mean = s[0] / T.float32(C)  # pyright: ignore[reportCallIssue]
                for i in T.Parallel(C):
                    x_frag[i] = x_frag[i] - mean
                    sq_frag[i] = x_frag[i] * x_frag[i]
                T.reduce_sum(sq_frag, s, dim=-1, clear=True)
                rstd = T.rsqrt(  # pyright: ignore[reportCallIssue]
                    s[0] / T.float32(C) + T.float32(_LN_EPS)  # pyright: ignore[reportCallIssue]
                )
                for i in T.Parallel(C):
                    x_frag[i] = x_frag[i] * rstd * T.cast(
                        ln_preW[i], "float32"
                    ) + T.cast(ln_preB[i], "float32")

                # keep the last row for the prev_x copy
                if n == LEN - 1:
                    for i in T.Parallel(C):
                        x_last[0, i] = T.cast(x_frag[i], DTYPE)

                # shift source: prev_x for the first token, else LN_pre(x0[n-1])
                if n == 0:
                    for i in T.Parallel(C):
                        out[n, i] = T.cast(
                            x_frag[i]
                            + T.cast(x_k[i], "float32")
                            * (T.cast(prev_x[i], "float32") - x_frag[i]),
                            DTYPE,
                        )
                else:
                    # recompute LN_pre(x0[n-1]) locally
                    s2 = T.alloc_fragment((1,), "float32")
                    p_frag = T.alloc_fragment((C,), "float32")
                    sq2_frag = T.alloc_fragment((C,), "float32")
                    T.copy(x0[n - 1, :], p_frag)
                    T.reduce_sum(p_frag, s2, dim=-1, clear=True)
                    mean2 = s2[0] / T.float32(C)  # pyright: ignore[reportCallIssue]
                    for i in T.Parallel(C):
                        p_frag[i] = p_frag[i] - mean2
                        sq2_frag[i] = p_frag[i] * p_frag[i]
                    T.reduce_sum(sq2_frag, s2, dim=-1, clear=True)
                    rstd2 = T.rsqrt(  # pyright: ignore[reportCallIssue]
                        s2[0] / T.float32(C) + T.float32(_LN_EPS)  # pyright: ignore[reportCallIssue]
                    )
                    for i in T.Parallel(C):
                        p_frag[i] = p_frag[i] * rstd2 * T.cast(
                            ln_preW[i], "float32"
                        ) + T.cast(ln_preB[i], "float32")
                    for i in T.Parallel(C):
                        out[n, i] = T.cast(
                            x_frag[i]
                            + T.cast(x_k[i], "float32") * (p_frag[i] - x_frag[i]),
                            DTYPE,
                        )

            # kernel 2: copy back to prev_x
            with T.Kernel(1, threads=THREADS):
                T.copy(x_last[0, :], prev_x)
        else:
            x_ln = T.alloc_global((LEN, C), DTYPE)

            # kernel 1: LN_pre over all tokens -> x_ln
            with T.Kernel(LEN, threads=THREADS) as n:
                ln(n, x0[n, :], ln_preW, ln_preB, x_ln=x_ln)

            # kernel 2: token-shift lerp (x_ln[n-1] already written by kernel 1)
            with T.Kernel(LEN, threads=THREADS) as n:
                for i in T.Parallel(C):
                    p_val = prev_x[i] if n == 0 else x_ln[n - 1, i]
                    out[n, i] = T.cast(
                        T.cast(x_ln[n, i], "float32")
                        + T.cast(x_k[i], "float32")
                        * (T.cast(p_val, "float32") - T.cast(x_ln[n, i], "float32")),
                        DTYPE,
                    )

            # kernel 3: copy back to prev_x
            with T.Kernel(1, threads=THREADS):
                T.copy(x_ln[LEN - 1, :], prev_x)

    return _impl


def cmix_prefill_main_macro(
    LEN, C: int, DTYPE: str, LEN_block: int, THREADS: int = 128
):
    """Fused cmix main: ``out = x0 + relusq(x @ kWt) @ vWt``.

    Args:
        LEN_block: sequence-tile size used to split the input into blocks for
            the kernel. It should be a multiple of 16 and chosen to match the
            typical sequence length for better occupancy.
        THREADS: threads per block for the two T.gemm kernels (warp partition
            for the MMA tile).
    """
    assert LEN_block % 16 == 0
    HID = 4 * C
    dtype_bytes = T.dtype(DTYPE).bytes  # pyright: ignore[reportCallIssue]

    # TODO: tune these and THREADS parameter
    _BK = 32
    STAGES = 2
    HID_block = 128

    # shared ≤ 48KB  (note: `^` is XOR, use `**` for power)
    assert (LEN_block * _BK + _BK * HID_block) * STAGES * dtype_bytes <= 48 * 2**10 # pyright: ignore[reportOperatorIssue]

    # # Look at https://github.com/tile-ai/tilelang/issues/2916
    # LEN = T.dynamic("LEN")
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
        """2 kernels."""
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


@tilelang.jit(out_idx=[7])
def cmix_prefill(C: int, DTYPE: str, LEN_block: int):
    HID = 4 * C

    LEN = T.dynamic("LEN")
    prologue = cmix_prefill_prologue_macro(LEN, C, DTYPE)
    main = cmix_prefill_main_macro(LEN, C, DTYPE, LEN_block)

    @T.prim_func
    def _impl(
        x0: T.Tensor((LEN, C), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        kWt: T.Tensor((C, HID), DTYPE),
        vWt: T.Tensor((HID, C), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        """5 kernels."""
        x = T.alloc_global((LEN, C), DTYPE)
        prologue(x0, ln_preW, ln_preB, x_k, prev_x=prev_x, out=x)

        # main (kernels 4-5)
        main(x, x0, kWt, vWt, out=out)

    return _impl
