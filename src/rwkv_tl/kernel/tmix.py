# pyright: reportInvalidTypeForm=false

import math

import tilelang
import tilelang.language as T

from ._common import HEAD_DIM, SERIAL, WARP
from .gemv import gemv_batch_macro, gemv_macro
from .ln import ln_prologue_macro

N = HEAD_DIM
_SQRT_E = math.sqrt(math.e)  # exp decay gate constant


def _gate_gemv_macro(R: int, C: int, DTYPE: str):
    """One low-rank first-step GEMV row: ``out[j] = sum_c W[j, c] * x[c]``.

    A warp per block reduces over ``C`` into one rank-row output (the
    ``[R, C]`` weight's rows each map to one ``out[j]``). One instance per
    gate (v/w/a/g) is bound to that gate's rank and its own weight/input/
    output; the rank-GEMV kernel launches all four in one 2D grid
    ``(rank row, gate)``.

    Args:
        R: Gate rank (row count of ``W`` / length of ``out``).
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
    """

    @T.macro
    def _impl(
        j,
        n,
        x: T.Tensor((C,), DTYPE),
        W: T.Tensor((R, C), DTYPE),
        *,
        out: T.Tensor((R,), DTYPE),
    ):
        """Args:
            j: Rank-row index (blockIdx.x); must be ``< R``.
            n: Thread index within the warp.
            x: This gate's shifted activation ``[C]`` (e.g. ``xv``).
            W: This gate's rank-in weight ``[R, C]`` (e.g. ``v1t``).
            out: This gate's rank-out result ``[R]`` (e.g. ``vr``).
        """
        acc = T.alloc_fragment((1,), "float32")
        acc[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
        for c in T.serial(C // WARP):
            c_idx = n * (C // WARP) + c
            acc[0] += T.cast(W[j, c_idx], "float32") * T.cast(x[c_idx], "float32")
        total = T.warp_reduce_sum(acc[0])
        if n == 0:
            out[j] = T.cast(total, DTYPE)

    return _impl


##### decode


def _tmix_shift6_macro(C: int, DTYPE: str):
    """Six token-shift lerps of one LN'd row: ``x_out = ln_val + w*(shift - ln_val)``.

    Consumes one row's LN output (``ln_val``, e.g. from ``ln_prologue_macro``)
    and the shift source (``shift``: previous token's LN output, or ``prev_x``
    for the first token), writing the six shifted activations to the stacked
    ``xrkv [3, C]`` (r/k/v) and ``xw``/``xa``/``xg`` ``[C]``. ``xv`` equals
    ``xrkv[2]`` and is written as its own ``[C]`` buffer for the low-rank gate
    kernel.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
    """

    @T.macro
    def _impl(
        ln_val: T.Tensor((C,), "float32"),
        shift: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        *,
        xrkv: T.Tensor((3, C), DTYPE),
        xv: T.Tensor((C,), DTYPE),
        xw: T.Tensor((C,), DTYPE),
        xa: T.Tensor((C,), DTYPE),
        xg: T.Tensor((C,), DTYPE),
    ):
        """Args:
            ln_val: Current row's LN_pre output ``[C]`` (fp32, before the
                DTYPE store).
            shift: Shift source ``[C]`` (prev token's LN output / ``prev_x``).
            x_rkvwag: Six token-shift weights stacked ``[6, C]`` (r/k/v/w/a/g).
            xrkv: Stacked r/k/v lerps ``[3, C]``.
            xv/xw/xa/xg: Low-rank gate inputs ``[C]``.
        """
        for i in T.Parallel(C):
            lv = ln_val[i]
            diff = T.cast(shift[i], "float32") - lv
            xrkv[0, i] = T.cast(lv + T.cast(x_rkvwag[0, i], "float32") * diff, DTYPE)
            xrkv[1, i] = T.cast(lv + T.cast(x_rkvwag[1, i], "float32") * diff, DTYPE)
            xrkv[2, i] = T.cast(lv + T.cast(x_rkvwag[2, i], "float32") * diff, DTYPE)
            xv[i] = xrkv[2, i]
            xw[i] = T.cast(lv + T.cast(x_rkvwag[3, i], "float32") * diff, DTYPE)
            xa[i] = T.cast(lv + T.cast(x_rkvwag[4, i], "float32") * diff, DTYPE)
            xg[i] = T.cast(lv + T.cast(x_rkvwag[5, i], "float32") * diff, DTYPE)

    return _impl


def tmix_decode_prologue_macro(C: int, DTYPE: str, THREADS: int = 256):
    """TMIX decode prologue: LN_pre + 6 token-shift lerps, prev_x updated.

    Fuses into ONE kernel: LN_pre leaves its normalized row in registers, the
    six token-shift lerps (xr/xw/xk/xv/xa/xg) consume it directly, and
    ``prev_x`` is overwritten in place with the raw LN output -- each thread
    reads and writes the same ``prev_x[i]`` element, so there is no
    cross-thread race. The six lerp weights arrive stacked as ``x_rkvwag``
    ``[6, C]`` (rows r/k/v/w/a/g). The r/k/v lerps are packed into one stacked
    ``xrkv`` ``[3, C]`` buffer (rows r/k/v) matching the stacked ``rkvWt``
    weight; the four low-rank gate inputs (``xv``/``xw``/``xa``/``xg``) are
    written as separate ``[C]`` buffers.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block; must divide ``C``.
    """

    # TODO: tune the THREADS parameter
    assert C % THREADS == 0

    ln = ln_prologue_macro(C, DTYPE)
    shift6 = _tmix_shift6_macro(C, DTYPE)

    @T.macro
    def _impl(
        x0: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        xrkv: T.Tensor((3, C), DTYPE),
        xv: T.Tensor((C,), DTYPE),
        xw: T.Tensor((C,), DTYPE),
        xa: T.Tensor((C,), DTYPE),
        xg: T.Tensor((C,), DTYPE),
    ):
        """1 kernel."""
        with T.Kernel(1, threads=THREADS):
            x_frag, rstd = ln(x0)
            ln_frag = T.alloc_fragment((C,), "float32")
            for i in T.Parallel(C):
                ln_val = x_frag[i] * rstd * T.cast(ln_preW[i], "float32") + T.cast(
                    ln_preB[i], "float32"
                )
                ln_frag[i] = ln_val
            shift6(ln_frag, prev_x, x_rkvwag, xrkv=xrkv, xv=xv, xw=xw, xa=xa, xg=xg)
            # store the LN row to prev_x AFTER the shift consumed it
            for i in T.Parallel(C):
                prev_x[i] = T.cast(ln_frag[i], DTYPE)

    return _impl


def tmix_decode_main_macro(
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
    THREADS: int = WARP,
):
    """TMIX decode main: rkv GEMVs + low-rank gates + DPLR + GN + out.

    The heavy tail of decode after the prologue's shifted activations: r/k/v
    projections, the four low-rank gate chains, the fused L2-norm + DPLR
    recurrence (state S fp32, updated in place), GroupNorm + r*k*r_k residual,
    and the ``g``-gated output projection with the ``x0`` residual.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        H: Head count (``C // N``).
        Rv/Rw/Ra/Rg: Rank of each low-rank gate.
        THREADS: threads per block for the rkv/oWt GEMVs; must divide ``C``.
    """
    assert THREADS % WARP == 0
    assert C % THREADS == 0
    assert H * N == C
    assert Rv % WARP == 0 and Rw % WARP == 0
    assert Ra % WARP == 0 and Rg % WARP == 0

    gv = gemv_batch_macro(C, C, 3, DTYPE, THREADS)  # [3, K, M] = [3, C, C]: xrkv @ rkvWt

    @T.macro
    def _impl(
        xrkv: T.Tensor((3, C), DTYPE),
        xv: T.Tensor((C,), DTYPE),
        xw: T.Tensor((C,), DTYPE),
        xa: T.Tensor((C,), DTYPE),
        xg: T.Tensor((C,), DTYPE),
        rkvWt: T.Tensor((3, C, C), DTYPE),
        v1t: T.Tensor((Rv, C), DTYPE),
        w1t: T.Tensor((Rw, C), DTYPE),
        a1t: T.Tensor((Ra, C), DTYPE),
        g1t: T.Tensor((Rg, C), DTYPE),
        v2t: T.Tensor((C, Rv), DTYPE),
        w2t: T.Tensor((C, Rw), DTYPE),
        a2t: T.Tensor((C, Ra), DTYPE),
        g2t: T.Tensor((C, Rg), DTYPE),
        v0: T.Tensor((C,), DTYPE),
        w0: T.Tensor((C,), DTYPE),
        a0: T.Tensor((C,), DTYPE),
        k_k: T.Tensor((C,), DTYPE),
        k_a: T.Tensor((C,), DTYPE),
        r_k: T.Tensor((H, N), DTYPE),
        ln_xW: T.Tensor((C,), DTYPE),
        ln_xB: T.Tensor((C,), DTYPE),
        oWt: T.Tensor((C, C), DTYPE),
        x0: T.Tensor((C,), DTYPE),
        *,
        rnn: T.Tensor((H, N, N), "float32"),
        v_first: T.Tensor((C,), DTYPE),
        first: T.int32,
        out: T.Tensor((C,), DTYPE),
    ):
        """Args:
        rnn: DPLR state ``[H, N, N]`` fp32; updated in place.
        v_first: v-residual gate state ``[C]``; on ``first != 0`` the gate
            is skipped (``v`` kept) and ``v_first`` is set to ``v``.
        first: nonzero on the first token of a sequence (``v_first`` has no
            prior value yet).
        out: output ``[C]`` = ``x0 + relusq... `` (full TMIX result).
        """
        rkv = T.alloc_global((3, C), DTYPE)
        vr = T.alloc_global((Rv,), DTYPE)
        wr = T.alloc_global((Rw,), DTYPE)
        ar = T.alloc_global((Ra,), DTYPE)
        gr = T.alloc_global((Rg,), DTYPE)
        w = T.alloc_global((C,), DTYPE)
        a = T.alloc_global((C,), DTYPE)
        kk = T.alloc_global((C,), DTYPE)
        kk_norm = T.alloc_global((C,), DTYPE)
        B = T.alloc_global((C,), DTYPE)
        y = T.alloc_global((C,), DTYPE)
        g = T.alloc_global((C,), DTYPE)
        gy = T.alloc_global((C,), DTYPE)

        # kernel: rkv = xrkv @ rkvWt (one batched GEMV over the 3 projections)
        with T.Kernel(C // THREADS, 3, threads=THREADS) as (bx, bz):
            gv(bx, bz, xrkv, rkvWt, out=rkv)

        # kernel: first-step rank GEMVs; 2D grid (rank row, gate) -- each
        # gate's block reads its own weight/input directly via its own macro
        # instance, no flat row-segment offsets.
        v_gate = _gate_gemv_macro(Rv, C, DTYPE)
        w_gate = _gate_gemv_macro(Rw, C, DTYPE)
        a_gate = _gate_gemv_macro(Ra, C, DTYPE)
        g_gate = _gate_gemv_macro(Rg, C, DTYPE)
        with T.Kernel(max(Rv, Rw, Ra, Rg), 4, threads=WARP) as (j, bz):
            n = T.get_thread_binding(0)
            if bz == 0:
                if j < Rv:
                    v_gate(j, n, xv, v1t, out=vr)
            elif bz == 1:
                if j < Rw:
                    w_gate(j, n, xw, w1t, out=wr)
            elif bz == 2:
                if j < Ra:
                    a_gate(j, n, xa, a1t, out=ar)
            else:
                if j < Rg:
                    g_gate(j, n, xg, g1t, out=gr)

        # kernel: rank-out second steps + v/w/a/kk/k/g gate math
        with T.Kernel(C, threads=WARP) as i:
            n = T.get_thread_binding(0)
            pv = T.alloc_fragment((1,), "float32")
            pw = T.alloc_fragment((1,), "float32")
            pa = T.alloc_fragment((1,), "float32")
            pg = T.alloc_fragment((1,), "float32")
            pv[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            pw[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            pa[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            pg[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(Rv // WARP):
                j_idx = n * (Rv // WARP) + j
                pv[0] += T.cast(v2t[i, j_idx], "float32") * T.cast(vr[j_idx], "float32")
            for j in T.serial(Rw // WARP):
                j_idx = n * (Rw // WARP) + j
                pw[0] += T.cast(w2t[i, j_idx], "float32") * T.tanh(
                    T.cast(wr[j_idx], "float32")
                )
            for j in T.serial(Ra // WARP):
                j_idx = n * (Ra // WARP) + j
                pa[0] += T.cast(a2t[i, j_idx], "float32") * T.cast(ar[j_idx], "float32")
            for j in T.serial(Rg // WARP):
                j_idx = n * (Rg // WARP) + j
                pg[0] += T.cast(g2t[i, j_idx], "float32") * T.sigmoid(
                    T.cast(gr[j_idx], "float32")
                )
            v12 = T.warp_reduce_sum(pv[0])
            w12 = T.warp_reduce_sum(pw[0])
            a12 = T.warp_reduce_sum(pa[0])
            g12 = T.warp_reduce_sum(pg[0])
            if n == 0:
                v_cur = T.cast(rkv[2, i], "float32")
                vf = T.if_then_else(first != 0, v_cur, T.cast(v_first[i], "float32"))
                sig_v = T.sigmoid(T.cast(v0[i], "float32") + v12)
                v_out = v_cur + sig_v * (vf - v_cur)
                # v gate state: first token stores v, later tokens keep v_first
                v_first[i] = T.cast(vf, DTYPE)
                rkv[2, i] = T.cast(v_out, DTYPE)
                w[i] = T.cast(
                    T.exp(
                        -T.sigmoid(T.cast(w0[i], "float32") + w12) / T.float32(_SQRT_E)  # pyright: ignore[reportCallIssue]
                    ),
                    DTYPE,
                )
                a_val = T.sigmoid(T.cast(a0[i], "float32") + a12)
                a[i] = T.cast(a_val, DTYPE)
                k_cur = T.cast(rkv[1, i], "float32")
                kk[i] = T.cast(k_cur * T.cast(k_k[i], "float32"), DTYPE)
                rkv[1, i] = T.cast(
                    k_cur + T.cast(k_a[i], "float32") * (k_cur * a_val - k_cur),
                    DTYPE,
                )
                # g gate: g = g2t @ sigmoid(gr), stored for the output projection
                g[i] = T.cast(g12, DTYPE)

        # kernel: fused L2-norm(kk) + neg*multiply -> kk_norm, B
        with T.Kernel(H, threads=WARP) as h:
            n = T.get_thread_binding(0)
            p_sq = T.alloc_fragment((1,), "float32")
            p_sq[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                kk_val = T.cast(kk[h * N + idx], "float32")
                p_sq[0] += kk_val * kk_val
            total_sq = T.warp_reduce_sum(p_sq[0])
            den = T.max(T.sqrt(total_sq), T.float32(1e-12))  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                flat = h * N + idx
                kk_norm[flat] = T.cast(T.cast(kk[flat], "float32") / den, DTYPE)
                B[flat] = T.cast(
                    -(T.cast(kk_norm[flat], "float32") * T.cast(a[flat], "float32")),
                    DTYPE,
                )

        # kernel: DPLR state update (one block per (h, v_n))
        with T.Kernel(H, N, threads=WARP) as (h, v_n):
            n = T.get_thread_binding(0)
            p_sa = T.alloc_fragment((1,), "float32")
            p_sa[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                a_idx = n * SERIAL + j
                p_sa[0] += rnn[h, v_n, a_idx] * T.cast(
                    kk_norm[h * N + a_idx], "float32"
                )
            sa = T.warp_reduce_sum(p_sa[0])
            v_val = T.cast(rkv[2, h * N + v_n], "float32")
            p_y = T.alloc_fragment((1,), "float32")
            p_y[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                k_idx = n * SERIAL + j
                s_new = (
                    rnn[h, v_n, k_idx] * T.cast(w[h * N + k_idx], "float32")
                    + sa * T.cast(B[h * N + k_idx], "float32")
                    + v_val * T.cast(rkv[1, h * N + k_idx], "float32")
                )
                rnn[h, v_n, k_idx] = s_new
                p_y[0] += s_new * T.cast(rkv[0, h * N + k_idx], "float32")
            y_val = T.warp_reduce_sum(p_y[0])
            if n == 0:
                y[h * N + v_n] = T.cast(y_val, DTYPE)

        # kernel: GroupNorm + r*k*r_k residual over [H, N]
        with T.Kernel(H, threads=WARP) as h:
            n = T.get_thread_binding(0)
            p_sum = T.alloc_fragment((1,), "float32")
            p_rkrk = T.alloc_fragment((1,), "float32")
            p_sum[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            p_rkrk[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                p_sum[0] += T.cast(y[h * N + idx], "float32")
                p_rkrk[0] += (
                    T.cast(rkv[0, h * N + idx], "float32")
                    * T.cast(rkv[1, h * N + idx], "float32")
                    * T.cast(r_k[h, idx], "float32")
                )
            total_sum = T.warp_reduce_sum(p_sum[0])
            total_rkrk = T.warp_reduce_sum(p_rkrk[0])
            mean = total_sum / T.float32(N)  # pyright: ignore[reportCallIssue]
            p_var = T.alloc_fragment((1,), "float32")
            p_var[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                diff = T.cast(y[h * N + idx], "float32") - mean
                p_var[0] += diff * diff
            total_var = T.warp_reduce_sum(p_var[0])
            rstd = T.float32(1.0) / T.sqrt(total_var / T.float32(N) + T.float32(64e-5))  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                flat = h * N + idx
                y_norm = (T.cast(y[flat], "float32") - mean) * rstd
                y_aff = y_norm * T.cast(ln_xW[flat], "float32") + T.cast(
                    ln_xB[flat], "float32"
                )
                residual = total_rkrk * T.cast(rkv[2, flat], "float32")
                gy[flat] = T.cast(
                    (y_aff + residual) * T.cast(g[flat], "float32"), DTYPE
                )

        # kernel: out = x0 + gy @ oWt
        gout = gemv_macro(C, C, DTYPE, THREADS)
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            gout(bx, gy, oWt, out=out)
        # add x0 residual
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            for i in T.Parallel(THREADS):
                idx = bx * THREADS + i
                if idx < C:
                    out[idx] = T.cast(
                        T.cast(out[idx], "float32") + T.cast(x0[idx], "float32"),
                        DTYPE,
                    )

    return _impl


@tilelang.jit(out_idx=[26])
def tmix_decode(
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
):
    """Fused single-token TMIX: LN_pre + 6-shift + rkv + gates + DPLR + GN + out.

    ``prev_x`` and ``rnn`` are updated in place; ``v_first`` carries the
    v-residual gate state across tokens (pass ``first=1`` on the first token).
    """
    prologue = tmix_decode_prologue_macro(C, DTYPE)
    main = tmix_decode_main_macro(C, DTYPE, H, Rv, Rw, Ra, Rg)

    @T.prim_func
    def _impl(
        x0: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        rkvWt: T.Tensor((3, C, C), DTYPE),
        v1t: T.Tensor((Rv, C), DTYPE),
        w1t: T.Tensor((Rw, C), DTYPE),
        a1t: T.Tensor((Ra, C), DTYPE),
        g1t: T.Tensor((Rg, C), DTYPE),
        v2t: T.Tensor((C, Rv), DTYPE),
        w2t: T.Tensor((C, Rw), DTYPE),
        a2t: T.Tensor((C, Ra), DTYPE),
        g2t: T.Tensor((C, Rg), DTYPE),
        v0: T.Tensor((C,), DTYPE),
        w0: T.Tensor((C,), DTYPE),
        a0: T.Tensor((C,), DTYPE),
        k_k: T.Tensor((C,), DTYPE),
        k_a: T.Tensor((C,), DTYPE),
        r_k: T.Tensor((H, N), DTYPE),
        ln_xW: T.Tensor((C,), DTYPE),
        ln_xB: T.Tensor((C,), DTYPE),
        oWt: T.Tensor((C, C), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        rnn: T.Tensor((H, N, N), "float32"),
        v_first: T.Tensor((C,), DTYPE),
        first: T.int32,
        out: T.Tensor((C,), DTYPE),
    ):
        xrkv = T.alloc_global((3, C), DTYPE)
        xv = T.alloc_global((C,), DTYPE)
        xw = T.alloc_global((C,), DTYPE)
        xa = T.alloc_global((C,), DTYPE)
        xg = T.alloc_global((C,), DTYPE)

        prologue(
            x0,
            ln_preW,
            ln_preB,
            x_rkvwag,
            prev_x=prev_x,
            xrkv=xrkv,
            xv=xv,
            xw=xw,
            xa=xa,
            xg=xg,
        )

        main(
            xrkv,
            xv,
            xw,
            xa,
            xg,
            rkvWt,
            v1t,
            w1t,
            a1t,
            g1t,
            v2t,
            w2t,
            a2t,
            g2t,
            v0,
            w0,
            a0,
            k_k,
            k_a,
            r_k,
            ln_xW,
            ln_xB,
            oWt,
            x0,
            rnn=rnn,
            v_first=v_first,
            first=first,
            out=out,
        )

    return _impl


##### prefill


def tmix_prefill_prologue_macro(LEN: int, C: int, DTYPE: str, THREADS: int = 256):
    """TMIX prefill prologue: LN_pre + 6 token-shift lerps over a sequence.

    Two kernels: kernel 1 computes every row's LN_pre output into ``x_ln``
    (``[LEN, C]``); kernel 2 applies the six token-shift lerps (weights stacked
    in ``x_rkvwag [6, C]``) using ``x_ln[t-1]`` as the shift source for
    ``t > 0`` and ``prev_x`` for ``t == 0`` -- the same stored-shift semantics
    as the reference, avoiding per-block recompute drift. ``prev_x`` is updated
    to the last row's LN output. The r/k/v lerps go to ``xrkv [3, LEN, C]``;
    the gate inputs ``xv``/``xw``/``xa``/``xg`` to ``[LEN, C]``.

    Args:
        LEN: Sequence length.
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block; must divide ``C``.
    """
    assert C % THREADS == 0

    ln = ln_prologue_macro(C, DTYPE)

    @T.macro
    def _impl(
        x0: T.Tensor((LEN, C), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        x_ln: T.Tensor((LEN, C), DTYPE),
        xr: T.Tensor((LEN, C), DTYPE),
        xk: T.Tensor((LEN, C), DTYPE),
        xv: T.Tensor((LEN, C), DTYPE),
        xw: T.Tensor((LEN, C), DTYPE),
        xa: T.Tensor((LEN, C), DTYPE),
        xg: T.Tensor((LEN, C), DTYPE),
    ):
        """2 kernels."""
        # kernel 1: LN_pre over all rows -> x_ln
        with T.Kernel(LEN, threads=THREADS) as t:
            x_frag, rstd = ln(x0[t, :])
            for i in T.Parallel(C):
                ln_val = x_frag[i] * rstd * T.cast(ln_preW[i], "float32") + T.cast(
                    ln_preB[i], "float32"
                )
                x_ln[t, i] = T.cast(ln_val, DTYPE)

        # kernel 2: six token-shift lerps; shift = prev_x (t=0) else x_ln[t-1]
        with T.Kernel(LEN, threads=THREADS) as t:
            if t == 0:
                for i in T.Parallel(C):
                    lv = T.cast(x_ln[t, i], "float32")
                    diff = T.cast(prev_x[i], "float32") - lv
                    xr[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[0, i], "float32") * diff, DTYPE
                    )
                    xk[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[1, i], "float32") * diff, DTYPE
                    )
                    xv[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[2, i], "float32") * diff, DTYPE
                    )
                    xw[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[3, i], "float32") * diff, DTYPE
                    )
                    xa[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[4, i], "float32") * diff, DTYPE
                    )
                    xg[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[5, i], "float32") * diff, DTYPE
                    )
            else:
                for i in T.Parallel(C):
                    lv = T.cast(x_ln[t, i], "float32")
                    diff = T.cast(x_ln[t - 1, i], "float32") - lv
                    xr[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[0, i], "float32") * diff, DTYPE
                    )
                    xk[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[1, i], "float32") * diff, DTYPE
                    )
                    xv[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[2, i], "float32") * diff, DTYPE
                    )
                    xw[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[3, i], "float32") * diff, DTYPE
                    )
                    xa[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[4, i], "float32") * diff, DTYPE
                    )
                    xg[t, i] = T.cast(
                        lv + T.cast(x_rkvwag[5, i], "float32") * diff, DTYPE
                    )
            if t == LEN - 1:
                for i in T.Parallel(C):
                    prev_x[i] = x_ln[t, i]

    return _impl


def _tmix_prefill_front_macro(
    LEN: int,
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
    THREADS: int = WARP,
):
    """TMIX prefill front: prologue + rkv GEMM + low-rank gates + L2-norm.

    Produces the per-row quantities the serial DPLR needs: ``rkv [3, LEN, C]``,
    ``w``/``a`` ``[LEN, C]``, ``kk_norm``/``B`` ``[LEN, C]``, and ``g``
    ``[LEN, C]``. Updates ``v_first`` in place.
    """
    Rsum = Rv + Rw + Ra + Rg
    assert C % THREADS == 0
    assert H * N == C
    assert C % WARP == 0
    assert Rv % WARP == 0 and Rw % WARP == 0
    assert Ra % WARP == 0 and Rg % WARP == 0


    prologue = tmix_prefill_prologue_macro(LEN, C, DTYPE)

    # Shared macro for the 4 first-step rank GEMVs ([LEN, C] @ [C, R]), used
    # as a @T.macro so each gate expands to its own kernel block. Weights are
    # stored transposed ``v1t [R, C]``, so B is read transposed into shared.
    BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 32, 128, 32, 2

    @T.macro
    def _rank_gemm(xs, Wt, out, R, col0):
        with T.Kernel(T.ceildiv(R, BLOCK_N), T.ceildiv(LEN, BLOCK_M), threads=128) as (
            bx,
            by,
        ):
            A_sh = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_sh = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_frag)
            for kk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=NUM_STAGES):
                T.copy(xs[by * BLOCK_M, kk * BLOCK_K], A_sh)
                for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                    if bx * BLOCK_N + j < R:
                        B_sh[i, j] = Wt[bx * BLOCK_N + j, kk * BLOCK_K + i]
                T.gemm(A_sh, B_sh, C_frag)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                if bx * BLOCK_N + j < R:
                    out[by * BLOCK_M + i, col0 + bx * BLOCK_N + j] = T.cast(
                        C_frag[i, j], DTYPE
                    )

    # rank-out GEMM: out [LEN, C] = act(rw[:, Roff:Roff+R]) @ Wt where Wt is the
    # rank-out weight [R, C] stored transposed as [C, R] (v2t/w2t/a2t/g2t), read
    # transposed into shared. ``activate`` selects tanh (w), sigmoid (g) or
    # identity (v/a) applied to the rw row before the matmul.
    @T.macro
    def _rank_out_gemm(rw, Wt, out, R, Roff, activate):
        with T.Kernel(T.ceildiv(C, BLOCK_N), T.ceildiv(LEN, BLOCK_M), threads=128) as (
            bx,
            by,
        ):
            A_sh = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_sh = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_frag)
            for kk in T.Pipelined(T.ceildiv(R, BLOCK_K), num_stages=NUM_STAGES):
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    v = T.cast(rw[by * BLOCK_M + i, Roff + kk * BLOCK_K + j], "float32")
                    if activate == "tanh":
                        A_sh[i, j] = T.cast(T.tanh(v), DTYPE)
                    elif activate == "sigmoid":
                        A_sh[i, j] = T.cast(T.sigmoid(v), DTYPE)
                    else:
                        A_sh[i, j] = T.cast(v, DTYPE)
                for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                    B_sh[i, j] = Wt[bx * BLOCK_N + j, kk * BLOCK_K + i]
                T.gemm(A_sh, B_sh, C_frag)
            T.copy(C_frag, out[by * BLOCK_M, bx * BLOCK_N])

    @T.macro
    def _impl(
        x0: T.Tensor((LEN, C), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        rkvWt: T.Tensor((3, C, C), DTYPE),
        v1t: T.Tensor((Rv, C), DTYPE),
        w1t: T.Tensor((Rw, C), DTYPE),
        a1t: T.Tensor((Ra, C), DTYPE),
        g1t: T.Tensor((Rg, C), DTYPE),
        v2t: T.Tensor((C, Rv), DTYPE),
        w2t: T.Tensor((C, Rw), DTYPE),
        a2t: T.Tensor((C, Ra), DTYPE),
        g2t: T.Tensor((C, Rg), DTYPE),
        v0: T.Tensor((C,), DTYPE),
        w0: T.Tensor((C,), DTYPE),
        a0: T.Tensor((C,), DTYPE),
        k_k: T.Tensor((C,), DTYPE),
        k_a: T.Tensor((C,), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        v_first: T.Tensor((LEN, C), DTYPE),
        first: T.int32,
        x_ln: T.Tensor((LEN, C), DTYPE),
        xr: T.Tensor((LEN, C), DTYPE),
        xk: T.Tensor((LEN, C), DTYPE),
        xv: T.Tensor((LEN, C), DTYPE),
        xw: T.Tensor((LEN, C), DTYPE),
        xa: T.Tensor((LEN, C), DTYPE),
        xg: T.Tensor((LEN, C), DTYPE),
        rkv: T.Tensor((3, LEN, C), DTYPE),
        w: T.Tensor((LEN, C), DTYPE),
        a: T.Tensor((LEN, C), DTYPE),
        kk_norm: T.Tensor((LEN, C), DTYPE),
        B: T.Tensor((LEN, C), DTYPE),
        g: T.Tensor((LEN, C), DTYPE),
    ):
        """5 + 3 kernels."""
        rw = T.alloc_global((LEN, Rsum), DTYPE)
        kk = T.alloc_global((LEN, C), DTYPE)
        v12 = T.alloc_global((LEN, C), DTYPE)
        w12 = T.alloc_global((LEN, C), DTYPE)
        a12 = T.alloc_global((LEN, C), DTYPE)
        g12 = T.alloc_global((LEN, C), DTYPE)

        prologue(
            x0,
            ln_preW,
            ln_preB,
            x_rkvwag,
            prev_x=prev_x,
            x_ln=x_ln,
            xr=xr,
            xk=xk,
            xv=xv,
            xw=xw,
            xa=xa,
            xg=xg,
        )

        # kernel: rkv = [xr; xk; xv] @ rkvWt  (batched GEMM over T rows x 3 projections)
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 16, 64, 32, 3
        with T.Kernel(T.ceildiv(C, BLOCK_N), T.ceildiv(LEN, BLOCK_M), 3, threads=128) as (
            bx,
            by,
            bz,
        ):
            A_sh = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_sh = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_frag)
            for k_blk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=NUM_STAGES):
                if bz == 0:
                    T.copy(xr[by * BLOCK_M, k_blk * BLOCK_K], A_sh)
                elif bz == 1:
                    T.copy(xk[by * BLOCK_M, k_blk * BLOCK_K], A_sh)
                else:
                    T.copy(xv[by * BLOCK_M, k_blk * BLOCK_K], A_sh)
                T.copy(rkvWt[bz, k_blk * BLOCK_K, bx * BLOCK_N], B_sh)
                T.gemm(A_sh, B_sh, C_frag)
            T.copy(C_frag, rkv[bz, by * BLOCK_M, bx * BLOCK_N])

        # kernel: first-step rank GEMVs as 4 independent packed T.gemm kernels
        # [LEN, C] @ [C, Rg] per gate, each writing its rw segment. Avoids
        # per-rank-row flat blocks that explode with larger ranks (1.5B: 5.6ms
        # flat -> ~0.47ms total GEMM).
        _rank_gemm(xv, v1t, rw, Rv, 0)
        _rank_gemm(xw, w1t, rw, Rw, Rv)
        _rank_gemm(xa, a1t, rw, Ra, Rv + Rw)
        _rank_gemm(xg, g1t, rw, Rg, Rv + Rw + Ra)

        # kernel: rank-out second steps as 4 packed T.gemm kernels
        # [LEN,R]@[R,C] per gate -> v12/w12/a12/g12 [LEN,C] (w/g rows activated
        # via tanh/sigmoid on the rw segments). Avoids LEN*C flat blocks that
        # explode with larger ranks (1.5B: 1.45ms flat -> ~0.33ms GEMM).
        _rank_out_gemm(rw, v2t, v12, Rv, 0, "none")
        _rank_out_gemm(rw, w2t, w12, Rw, Rv, "tanh")
        _rank_out_gemm(rw, a2t, a12, Ra, Rv + Rw, "none")
        _rank_out_gemm(rw, g2t, g12, Rg, Rv + Rw + Ra, "sigmoid")

        # kernel: v/w/a/kk/k/g gate math from v12/w12/a12/g12 + rkv; one block
        # per WARP C-elements (lane n -> element base+n), no cross-lane reduce.
        with T.Kernel(LEN * (C // WARP), threads=WARP) as flat:
            n = T.get_thread_binding(0)
            t = flat // (C // WARP)
            i = (flat % (C // WARP)) * WARP + n
            v12_v = T.cast(v12[t, i], "float32")
            w12_v = T.cast(w12[t, i], "float32")
            a12_v = T.cast(a12[t, i], "float32")
            g12_v = T.cast(g12[t, i], "float32")
            v_cur = T.cast(rkv[2, t, i], "float32")
            # v_first is the value residual from layer 0: layer 0
            # (``first != 0``) keeps v and stores its whole [T, C] v;
            # later layers gate each row toward layer-0's same-row v.
            vf = T.if_then_else(
                first != 0, v_cur, T.cast(v_first[t, i], "float32")
            )
            sig_v = T.sigmoid(T.cast(v0[i], "float32") + v12_v)
            v_out = v_cur + sig_v * (vf - v_cur)
            v_first[t, i] = T.cast(vf, DTYPE)
            rkv[2, t, i] = T.cast(v_out, DTYPE)
            w[t, i] = T.cast(
                T.exp(
                    -T.sigmoid(T.cast(w0[i], "float32") + w12_v) / T.float32(_SQRT_E)  # pyright: ignore[reportCallIssue]
                ),
                DTYPE,
            )
            a_val = T.sigmoid(T.cast(a0[i], "float32") + a12_v)
            a[t, i] = T.cast(a_val, DTYPE)
            k_cur = T.cast(rkv[1, t, i], "float32")
            kk[t, i] = T.cast(k_cur * T.cast(k_k[i], "float32"), DTYPE)
            rkv[1, t, i] = T.cast(
                k_cur
                + T.cast(k_a[i], "float32") * (k_cur * a_val - k_cur),
                DTYPE,
            )
            g[t, i] = T.cast(g12_v, DTYPE)

        # kernel: fused L2-norm(kk) + neg*multiply -> kk_norm, B; flat (t, h)
        with T.Kernel(LEN * H, threads=WARP) as flat:
            n = T.get_thread_binding(0)
            t = flat // H
            h = flat % H
            p_sq = T.alloc_fragment((1,), "float32")
            p_sq[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                kk_val = T.cast(kk[t, h * N + idx], "float32")
                p_sq[0] += kk_val * kk_val
            total_sq = T.warp_reduce_sum(p_sq[0])
            den = T.max(T.sqrt(total_sq), T.float32(1e-12))  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                flat_c = h * N + idx
                kk_norm[t, flat_c] = T.cast(
                    T.cast(kk[t, flat_c], "float32") / den, DTYPE
                )
                B[t, flat_c] = T.cast(
                    -(T.cast(kk_norm[t, flat_c], "float32") * T.cast(a[t, flat_c], "float32")),
                    DTYPE,
                )

    return _impl


@tilelang.jit(out_idx=[21, 22, 23, 24, 25, 26])
def _tmix_prefill_front(
    LEN: int,
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
):
    """Prefill front jit: returns (rkv, w, a, kk_norm, B, g) + updates state."""
    front = _tmix_prefill_front_macro(
        LEN, C, DTYPE, H, Rv, Rw, Ra, Rg
    )

    @T.prim_func
    def _impl(
        x0: T.Tensor((LEN, C), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_rkvwag: T.Tensor((6, C), DTYPE),
        rkvWt: T.Tensor((3, C, C), DTYPE),
        v1t: T.Tensor((Rv, C), DTYPE),
        w1t: T.Tensor((Rw, C), DTYPE),
        a1t: T.Tensor((Ra, C), DTYPE),
        g1t: T.Tensor((Rg, C), DTYPE),
        v2t: T.Tensor((C, Rv), DTYPE),
        w2t: T.Tensor((C, Rw), DTYPE),
        a2t: T.Tensor((C, Ra), DTYPE),
        g2t: T.Tensor((C, Rg), DTYPE),
        v0: T.Tensor((C,), DTYPE),
        w0: T.Tensor((C,), DTYPE),
        a0: T.Tensor((C,), DTYPE),
        k_k: T.Tensor((C,), DTYPE),
        k_a: T.Tensor((C,), DTYPE),
        prev_x: T.Tensor((C,), DTYPE),
        v_first: T.Tensor((LEN, C), DTYPE),
        first: T.int32,
        rkv: T.Tensor((3, LEN, C), DTYPE),
        w: T.Tensor((LEN, C), DTYPE),
        a: T.Tensor((LEN, C), DTYPE),
        kk_norm: T.Tensor((LEN, C), DTYPE),
        B: T.Tensor((LEN, C), DTYPE),
        g: T.Tensor((LEN, C), DTYPE),
    ):
        x_ln = T.alloc_global((LEN, C), DTYPE)
        xr = T.alloc_global((LEN, C), DTYPE)
        xk = T.alloc_global((LEN, C), DTYPE)
        xv = T.alloc_global((LEN, C), DTYPE)
        xw = T.alloc_global((LEN, C), DTYPE)
        xa = T.alloc_global((LEN, C), DTYPE)
        xg = T.alloc_global((LEN, C), DTYPE)

        front(
            x0,
            ln_preW,
            ln_preB,
            x_rkvwag,
            rkvWt,
            v1t,
            w1t,
            a1t,
            g1t,
            v2t,
            w2t,
            a2t,
            g2t,
            v0,
            w0,
            a0,
            k_k,
            k_a,
            prev_x=prev_x,
            v_first=v_first,
            first=first,
            x_ln=x_ln,
            xr=xr,
            xk=xk,
            xv=xv,
            xw=xw,
            xa=xa,
            xg=xg,
            rkv=rkv,
            w=w,
            a=a,
            kk_norm=kk_norm,
            B=B,
            g=g,
        )

    return _impl


def tmix_prefill_back_macro(
    LEN: int,
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
    THREADS: int = WARP,
):
    """TMIX prefill back: serial DPLR + GroupNorm + output projection.

    DPLR is serial over T (state-dependent); GN and the oWt projection are
    batched over T (one block per (t, h) and a `T.gemm` over [LEN, C],
    respectively) so they stay parallel.
    """
    assert C % THREADS == 0
    assert H * N == C
    assert C % WARP == 0
    assert Rv % WARP == 0 and Rw % WARP == 0
    assert Ra % WARP == 0 and Rg % WARP == 0

    @T.macro
    def _impl(
        rkv: T.Tensor((3, LEN, C), DTYPE),
        w: T.Tensor((LEN, C), DTYPE),
        kk_norm: T.Tensor((LEN, C), DTYPE),
        B: T.Tensor((LEN, C), DTYPE),
        a: T.Tensor((LEN, C), DTYPE),
        g: T.Tensor((LEN, C), DTYPE),
        r_k: T.Tensor((H, N), DTYPE),
        ln_xW: T.Tensor((C,), DTYPE),
        ln_xB: T.Tensor((C,), DTYPE),
        oWt: T.Tensor((C, C), DTYPE),
        x0: T.Tensor((LEN, C), DTYPE),
        *,
        rnn: T.Tensor((H, N, N), "float32"),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        """3 kernels."""
        y = T.alloc_global((LEN, C), DTYPE)
        gy = T.alloc_global((LEN, C), DTYPE)

        # kernel: DPLR state update, one block per head h with N threads
        # (thread = state column v_n); each column's N-element state lives in
        # per-thread registers (fp16, matching faster3a), so the serial over T
        # loop has no global/smem round-trip. Column updates are independent,
        # no cross-thread reduce.
        with T.Kernel(H, threads=N) as h:
            n = T.get_thread_binding(0)
            st = T.alloc_local((N,), "float16")
            for k in T.serial(N):
                st[k] = T.cast(rnn[h, n, k], "float16")

            for t in T.serial(LEN):
                p_sa = T.alloc_fragment((1,), "float32")
                p_y = T.alloc_fragment((1,), "float32")
                p_sa[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
                p_y[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
                for k in T.serial(N):
                    p_sa[0] += T.cast(st[k], "float32") * T.cast(
                        kk_norm[t, h * N + k], "float32"
                    )
                sa = p_sa[0]
                v_val = T.cast(rkv[2, t, h * N + n], "float32")
                for k in T.serial(N):
                    s_new = (
                        T.cast(st[k], "float32") * T.cast(w[t, h * N + k], "float32")
                        + sa * T.cast(B[t, h * N + k], "float32")
                        + v_val * T.cast(rkv[1, t, h * N + k], "float32")
                    )
                    st[k] = T.cast(s_new, "float16")
                    p_y[0] += s_new * T.cast(rkv[0, t, h * N + k], "float32")
                y[t, h * N + n] = T.cast(p_y[0], DTYPE)

            for k in T.serial(N):
                rnn[h, n, k] = T.cast(st[k], "float32")

        # kernel: GroupNorm + r*k*r_k residual over [H, N]; one block per (t, h)
        with T.Kernel(LEN, H, threads=WARP) as (t2, h2):
            n = T.get_thread_binding(0)
            p_sum = T.alloc_fragment((1,), "float32")
            p_rkrk = T.alloc_fragment((1,), "float32")
            p_sum[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            p_rkrk[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                p_sum[0] += T.cast(y[t2, h2 * N + idx], "float32")
                p_rkrk[0] += (
                    T.cast(rkv[0, t2, h2 * N + idx], "float32")
                    * T.cast(rkv[1, t2, h2 * N + idx], "float32")
                    * T.cast(r_k[h2, idx], "float32")
                )
            total_sum = T.warp_reduce_sum(p_sum[0])
            total_rkrk = T.warp_reduce_sum(p_rkrk[0])
            mean = total_sum / T.float32(N)  # pyright: ignore[reportCallIssue]
            p_var = T.alloc_fragment((1,), "float32")
            p_var[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                diff = T.cast(y[t2, h2 * N + idx], "float32") - mean
                p_var[0] += diff * diff
            total_var = T.warp_reduce_sum(p_var[0])
            rstd = T.float32(1.0) / T.sqrt(total_var / T.float32(N) + T.float32(64e-5))  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                idx = n * SERIAL + j
                flat_c = h2 * N + idx
                y_norm = (T.cast(y[t2, flat_c], "float32") - mean) * rstd
                y_aff = y_norm * T.cast(ln_xW[flat_c], "float32") + T.cast(
                    ln_xB[flat_c], "float32"
                )
                residual = total_rkrk * T.cast(rkv[2, t2, flat_c], "float32")
                gy[t2, flat_c] = T.cast(
                    (y_aff + residual) * T.cast(g[t2, flat_c], "float32"), DTYPE
                )

        # kernel: out = x0 + gy @ oWt  (batched T.gemm over [LEN, C] x [C, C])
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 16, 64, 32, 3
        with T.Kernel(T.ceildiv(C, BLOCK_N), T.ceildiv(LEN, BLOCK_M), threads=128) as (
            bx,
            by,
        ):
            A_sh = T.alloc_shared((BLOCK_M, BLOCK_K), DTYPE)
            B_sh = T.alloc_shared((BLOCK_K, BLOCK_N), DTYPE)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_frag)
            for kk in T.Pipelined(T.ceildiv(C, BLOCK_K), num_stages=NUM_STAGES):
                T.copy(gy[by * BLOCK_M, kk * BLOCK_K], A_sh)
                T.copy(oWt[kk * BLOCK_K, bx * BLOCK_N], B_sh)
                T.gemm(A_sh, B_sh, C_frag)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                out[by * BLOCK_M + i, bx * BLOCK_N + j] = T.cast(
                    C_frag[i, j]
                    + T.cast(x0[by * BLOCK_M + i, bx * BLOCK_N + j], "float32"),
                    DTYPE,
                )

    return _impl


@tilelang.jit(out_idx=[12])
def _tmix_prefill_back(
    LEN: int,
    C: int,
    DTYPE: str,
    H: int,
    Rv: int,
    Rw: int,
    Ra: int,
    Rg: int,
):
    """Prefill back jit: DPLR + GN + output projection."""
    back = tmix_prefill_back_macro(LEN, C, DTYPE, H, Rv, Rw, Ra, Rg)

    @T.prim_func
    def _impl(
        rkv: T.Tensor((3, LEN, C), DTYPE),
        w: T.Tensor((LEN, C), DTYPE),
        kk_norm: T.Tensor((LEN, C), DTYPE),
        B: T.Tensor((LEN, C), DTYPE),
        a: T.Tensor((LEN, C), DTYPE),
        g: T.Tensor((LEN, C), DTYPE),
        r_k: T.Tensor((H, N), DTYPE),
        ln_xW: T.Tensor((C,), DTYPE),
        ln_xB: T.Tensor((C,), DTYPE),
        oWt: T.Tensor((C, C), DTYPE),
        x0: T.Tensor((LEN, C), DTYPE),
        rnn: T.Tensor((H, N, N), "float32"),
        out: T.Tensor((LEN, C), DTYPE),
    ):
        back(
            rkv,
            w,
            kk_norm,
            B,
            a,
            g,
            r_k,
            ln_xW,
            ln_xB,
            oWt,
            x0,
            rnn=rnn,
            out=out,
        )

    return _impl
