# pyright: reportInvalidTypeForm=false

import math

import tilelang
import tilelang.language as T

from ._common import HEAD_DIM, SERIAL, WARP
from .gemv import gemv_macro

N = HEAD_DIM
_SQRT_E = math.sqrt(math.e)  # exp decay gate constant


##### decode


def tmix_decode_prologue_macro(C: int, DTYPE: str, THREADS: int = 256):
    """TMIX decode prologue: LN_pre + 6 token-shift lerps, prev_x updated.

    Fuses into ONE kernel: LN_pre leaves its normalized row in registers, the
    six token-shift lerps (xr/xw/xk/xv/xa/xg) consume it directly, and
    ``prev_x`` is overwritten in place with the raw LN output -- each thread
    reads and writes the same ``prev_x[i]`` element, so there is no
    cross-thread race.

    Args:
        C: Channel width.
        DTYPE: Element type, ``"float16"`` or ``"bfloat16"``.
        THREADS: threads per block; must divide ``C``.
    """

    # TODO: tune the THREADS parameter
    assert C % THREADS == 0

    @T.macro
    def _impl(
        x0: T.Tensor((C,), DTYPE),
        ln_preW: T.Tensor((C,), DTYPE),
        ln_preB: T.Tensor((C,), DTYPE),
        x_r: T.Tensor((C,), DTYPE),
        x_w: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        x_v: T.Tensor((C,), DTYPE),
        x_a: T.Tensor((C,), DTYPE),
        x_g: T.Tensor((C,), DTYPE),
        *,
        prev_x: T.Tensor((C,), DTYPE),
        xr: T.Tensor((C,), DTYPE),
        xw: T.Tensor((C,), DTYPE),
        xk: T.Tensor((C,), DTYPE),
        xv: T.Tensor((C,), DTYPE),
        xa: T.Tensor((C,), DTYPE),
        xg: T.Tensor((C,), DTYPE),
    ):
        """1 kernel."""
        with T.Kernel(1, threads=THREADS):
            s = T.alloc_fragment((1,), "float32")
            x_frag = T.alloc_fragment((C,), "float32")
            sq_frag = T.alloc_fragment((C,), "float32")

            T.copy(x0, x_frag)

            T.reduce_sum(x_frag, s, dim=-1, clear=True)
            mean = s[0] / T.float32(C)  # pyright: ignore[reportCallIssue]

            for i in T.Parallel(C):
                x_frag[i] = x_frag[i] - mean
                sq_frag[i] = x_frag[i] * x_frag[i]

            T.reduce_sum(sq_frag, s, dim=-1, clear=True)
            rstd = T.rsqrt(  # pyright: ignore[reportCallIssue]
                s[0] / T.float32(C) + T.float32(1e-5)  # pyright: ignore[reportCallIssue]
            )

            for i in T.Parallel(C):
                ln_val = x_frag[i] * rstd * T.cast(ln_preW[i], "float32") + T.cast(
                    ln_preB[i], "float32"
                )
                diff = T.cast(prev_x[i], "float32") - ln_val
                prev_x[i] = T.cast(ln_val, DTYPE)
                xr[i] = T.cast(ln_val + T.cast(x_r[i], "float32") * diff, DTYPE)
                xw[i] = T.cast(ln_val + T.cast(x_w[i], "float32") * diff, DTYPE)
                xk[i] = T.cast(ln_val + T.cast(x_k[i], "float32") * diff, DTYPE)
                xv[i] = T.cast(ln_val + T.cast(x_v[i], "float32") * diff, DTYPE)
                xa[i] = T.cast(ln_val + T.cast(x_a[i], "float32") * diff, DTYPE)
                xg[i] = T.cast(ln_val + T.cast(x_g[i], "float32") * diff, DTYPE)

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
        THREADS: threads per block for the GEMV/elementwise kernels; must
            divide ``C``, ``H*N`` and each rank.
    """
    assert C % THREADS == 0
    assert H * N == C
    assert Rv % THREADS == 0 and Rw % THREADS == 0
    assert Ra % THREADS == 0 and Rg % THREADS == 0

    gv = gemv_macro(C, C, DTYPE, THREADS)  # [K, M] = [C, C]: x @ rkvWt[b]

    @T.macro
    def _impl(
        xr: T.Tensor((C,), DTYPE),
        xk: T.Tensor((C,), DTYPE),
        xv: T.Tensor((C,), DTYPE),
        xw: T.Tensor((C,), DTYPE),
        xa: T.Tensor((C,), DTYPE),
        xg: T.Tensor((C,), DTYPE),
        rWt: T.Tensor((C, C), DTYPE),
        kWt: T.Tensor((C, C), DTYPE),
        vWt: T.Tensor((C, C), DTYPE),
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
        r = T.alloc_global((C,), DTYPE)
        k = T.alloc_global((C,), DTYPE)
        v = T.alloc_global((C,), DTYPE)
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

        # kernel: r = xr @ rWt; k = xk @ kWt; v = xv @ vWt
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            gv(bx, xr, rWt, out=r)
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            gv(bx, xk, kWt, out=k)
        with T.Kernel(C // THREADS, threads=THREADS) as bx:
            gv(bx, xv, vWt, out=v)

        # kernel: packed first-step rank GEMVs (one warp per rank row)
        with T.Kernel(Rv + Rw + Ra + Rg, threads=WARP) as (j,):
            n = T.get_thread_binding(0)
            acc = T.alloc_fragment((1,), "float32")
            acc[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for c in T.serial(C // WARP):
                c_idx = n * (C // WARP) + c
                xsel = T.if_then_else(
                    j < Rv,
                    xv[c_idx],
                    T.if_then_else(
                        j < Rv + Rw,
                        xw[c_idx],
                        T.if_then_else(j < Rv + Rw + Ra, xa[c_idx], xg[c_idx]),
                    ),
                )
                wsel = T.if_then_else(
                    j < Rv,
                    v1t[j, c_idx],
                    T.if_then_else(
                        j < Rv + Rw,
                        w1t[j - Rv, c_idx],
                        T.if_then_else(
                            j < Rv + Rw + Ra,
                            a1t[j - Rv - Rw, c_idx],
                            g1t[j - Rv - Rw - Ra, c_idx],
                        ),
                    ),
                )
                acc[0] += T.cast(wsel, "float32") * T.cast(xsel, "float32")
            total = T.warp_reduce_sum(acc[0])
            if n == 0:
                if j < Rv:
                    vr[j] = T.cast(total, DTYPE)
                elif j < Rv + Rw:
                    wr[j - Rv] = T.cast(total, DTYPE)
                elif j < Rv + Rw + Ra:
                    ar[j - Rv - Rw] = T.cast(total, DTYPE)
                else:
                    gr[j - Rv - Rw - Ra] = T.cast(total, DTYPE)

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
                v_cur = T.cast(v[i], "float32")
                vf = T.if_then_else(first != 0, v_cur, T.cast(v_first[i], "float32"))
                sig_v = T.sigmoid(T.cast(v0[i], "float32") + v12)
                v_out = v_cur + sig_v * (vf - v_cur)
                # v gate state: first token stores v, later tokens keep v_first
                v_first[i] = T.cast(vf, DTYPE)
                v[i] = T.cast(v_out, DTYPE)
                w[i] = T.cast(
                    T.exp(
                        -T.sigmoid(T.cast(w0[i], "float32") + w12) / T.float32(_SQRT_E)  # pyright: ignore[reportCallIssue]
                    ),
                    DTYPE,
                )
                a_val = T.sigmoid(T.cast(a0[i], "float32") + a12)
                a[i] = T.cast(a_val, DTYPE)
                kk[i] = T.cast(
                    T.cast(k[i], "float32") * T.cast(k_k[i], "float32"), DTYPE
                )
                k[i] = T.cast(
                    T.cast(k[i], "float32")
                    + T.cast(k_a[i], "float32")
                    * (T.cast(k[i], "float32") * a_val - T.cast(k[i], "float32")),
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
            v_val = T.cast(v[h * N + v_n], "float32")
            p_y = T.alloc_fragment((1,), "float32")
            p_y[0] = T.float32(0.0)  # pyright: ignore[reportCallIssue]
            for j in T.serial(SERIAL):
                k_idx = n * SERIAL + j
                s_new = (
                    rnn[h, v_n, k_idx] * T.cast(w[h * N + k_idx], "float32")
                    + sa * T.cast(B[h * N + k_idx], "float32")
                    + v_val * T.cast(k[h * N + k_idx], "float32")
                )
                rnn[h, v_n, k_idx] = s_new
                p_y[0] += s_new * T.cast(r[h * N + k_idx], "float32")
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
                    T.cast(r[h * N + idx], "float32")
                    * T.cast(k[h * N + idx], "float32")
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
                residual = total_rkrk * T.cast(v[flat], "float32")
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


@tilelang.jit(out_idx=[33])
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
        x_r: T.Tensor((C,), DTYPE),
        x_w: T.Tensor((C,), DTYPE),
        x_k: T.Tensor((C,), DTYPE),
        x_v: T.Tensor((C,), DTYPE),
        x_a: T.Tensor((C,), DTYPE),
        x_g: T.Tensor((C,), DTYPE),
        rWt: T.Tensor((C, C), DTYPE),
        kWt: T.Tensor((C, C), DTYPE),
        vWt: T.Tensor((C, C), DTYPE),
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
        xr = T.alloc_global((C,), DTYPE)
        xw = T.alloc_global((C,), DTYPE)
        xk = T.alloc_global((C,), DTYPE)
        xv = T.alloc_global((C,), DTYPE)
        xa = T.alloc_global((C,), DTYPE)
        xg = T.alloc_global((C,), DTYPE)

        prologue(
            x0,
            ln_preW,
            ln_preB,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            prev_x=prev_x,
            xr=xr,
            xw=xw,
            xk=xk,
            xv=xv,
            xa=xa,
            xg=xg,
        )

        main(
            xr,
            xk,
            xv,
            xw,
            xa,
            xg,
            rWt,
            kWt,
            vWt,
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
