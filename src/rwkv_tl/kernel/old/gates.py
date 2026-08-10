"""Fused gate kernels (elementwise over the embedding dim C), dtype-parameterized.

All gate math (sigmoid / exp / LERP) runs in fp32 and is cast to DTYPE only at
the store, favouring throughput over bit-exactness with PyTorch eager. C is a
model constant baked at compile time (compiled per-C, cached).

``build(DTYPE)`` returns a namespace bound to one element type.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
# NOTE: no `from __future__ import annotations` here -- tilelang's eager builder
# evaluates the annotation expressions, and a stringified annotation would lose
# the closure DTYPE param (NameError).

import math
from types import SimpleNamespace

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

from ._common import BLOCK, WARP

_SQRT_E = math.sqrt(math.e)  # exp decay gate constant


def build(DTYPE: str) -> SimpleNamespace:
    """Build the dtype-bound gate kernel namespace.

    Args:
        DTYPE: Element type string, ``"float16"`` or ``"bfloat16"``.

    Returns:
        ``SimpleNamespace`` with ``fused_w_gate``, ``fused_v_gate``,
        ``fused_a_kk_k``.
    """

    @tilelang.jit(out_idx=[2])
    def _w_gate_kernel(C: int):
        @T.prim_func
        def _impl(
            x: T.Tensor((C,), DTYPE),
            w0: T.Tensor((C,), DTYPE),
            out: T.Tensor((C,), DTYPE),
        ):
            """Fused w decay gate: w = exp(-sigmoid(w0 + x) / sqrt(e))."""
            for bx in T.thread_binding(  # type: ignore[operator]
                (C + BLOCK - 1) // BLOCK, "blockIdx.x"
            ):
                for tx in T.thread_binding(BLOCK, "threadIdx.x"):
                    i = bx * BLOCK + tx
                    if i < C:
                        s = T.cast(x[i], "float32") + T.cast(w0[i], "float32")
                        out[i] = T.cast(
                            T.exp(-T.sigmoid(s) / T.float32(_SQRT_E)), DTYPE
                        )

        return _impl

    @tilelang.jit(out_idx=[4])
    def _v_gate_kernel(C: int):
        @T.prim_func
        def _impl(
            v: T.Tensor((C,), DTYPE),
            v_first: T.Tensor((C,), DTYPE),
            v0: T.Tensor((C,), DTYPE),
            v12: T.Tensor((C,), DTYPE),
            out: T.Tensor((C,), DTYPE),
        ):
            """Fused v residual gate: v + sigmoid(v0 + v12) * (v_first - v)."""
            for bx in T.thread_binding(  # type: ignore[operator]
                (C + BLOCK - 1) // BLOCK, "blockIdx.x"
            ):
                for tx in T.thread_binding(BLOCK, "threadIdx.x"):
                    i = bx * BLOCK + tx
                    if i < C:
                        vf = T.cast(v[i], "float32")
                        sig = T.sigmoid(
                            T.cast(v0[i], "float32") + T.cast(v12[i], "float32")
                        )
                        out[i] = T.cast(
                            vf + sig * (T.cast(v_first[i], "float32") - vf), DTYPE
                        )

        return _impl

    @tilelang.jit(out_idx=[5, 6, 7])
    def _a_kk_k_kernel(C: int):
        @T.prim_func
        def _impl(
            a0: T.Tensor((C,), DTYPE),
            a_x: T.Tensor((C,), DTYPE),
            k: T.Tensor((C,), DTYPE),
            k_k: T.Tensor((C,), DTYPE),
            k_a: T.Tensor((C,), DTYPE),
            a_out: T.Tensor((C,), DTYPE),
            kk_out: T.Tensor((C,), DTYPE),
            k_out: T.Tensor((C,), DTYPE),
        ):
            """Fused a-gate + kk + k LERP.

            a = sigmoid(a0 + a_x); kk = k * k_k; new_k = k + k_a * (k * a - k).
            """
            for bx in T.thread_binding(  # type: ignore[operator]
                (C + BLOCK - 1) // BLOCK, "blockIdx.x"
            ):
                for tx in T.thread_binding(BLOCK, "threadIdx.x"):
                    i = bx * BLOCK + tx
                    if i < C:
                        kf = T.cast(k[i], "float32")
                        a_val = T.sigmoid(
                            T.cast(a0[i], "float32") + T.cast(a_x[i], "float32")
                        )
                        a_out[i] = T.cast(a_val, DTYPE)
                        kk_out[i] = T.cast(kf * T.cast(k_k[i], "float32"), DTYPE)
                        k_out[i] = T.cast(
                            kf + T.cast(k_a[i], "float32") * (kf * a_val - kf), DTYPE
                        )

        return _impl

    @tilelang.jit(out_idx=[16, 17, 18, 19, 20, 21])
    def _gates_kernel(Rv: int, Rw: int, Ra: int, Rg: int, C: int):
        @T.prim_func
        def _impl(
            vr: T.Tensor((Rv,), DTYPE),
            wr: T.Tensor((Rw,), DTYPE),
            ar: T.Tensor((Ra,), DTYPE),
            gr: T.Tensor((Rg,), DTYPE),
            v2t: T.Tensor((C, Rv), DTYPE),
            w2t: T.Tensor((C, Rw), DTYPE),
            a2t: T.Tensor((C, Ra), DTYPE),
            g2t: T.Tensor((C, Rg), DTYPE),
            v: T.Tensor((C,), DTYPE),
            v_first: T.Tensor((C,), DTYPE),
            k: T.Tensor((C,), DTYPE),
            v0: T.Tensor((C,), DTYPE),
            w0: T.Tensor((C,), DTYPE),
            a0: T.Tensor((C,), DTYPE),
            k_k: T.Tensor((C,), DTYPE),
            k_a: T.Tensor((C,), DTYPE),
            v_out: T.Tensor((C,), DTYPE),
            w_out: T.Tensor((C,), DTYPE),
            a_out: T.Tensor((C,), DTYPE),
            kk_out: T.Tensor((C,), DTYPE),
            k_out: T.Tensor((C,), DTYPE),
            g_out: T.Tensor((C,), DTYPE),
        ):
            """Fused low-rank gate second steps + gate math.

            Per output element i over C:
              v12 = sum_j v2t[i,j] * vr[j]        (v gate, rank Rv)
              w12 = sum_j w2t[i,j] * tanh(wr[j])  (w decay gate, rank Rw)
              a12 = sum_j a2t[i,j] * ar[j]        (a gate, rank Ra)
              g12 = sum_j g2t[i,j] * sigmoid(gr[j]) (g gate, rank Rg)
            then v/w/a/kk/k are finalized elementwise and g_out = g12.
            Merges the 4 rank-out matmuls + activations + the v/w/a gate
            elementwise into one launch. Each gate's rank is a separate
            compile-time constant (RWKV7 checkpoints use distinct ranks).
            """
            with T.Kernel(C, threads=WARP) as (i,):
                n = T.get_thread_binding(0)
                pv = T.alloc_fragment((1,), "float32")
                pw = T.alloc_fragment((1,), "float32")
                pa = T.alloc_fragment((1,), "float32")
                pg = T.alloc_fragment((1,), "float32")
                pv[0] = T.float32(0.0)
                pw[0] = T.float32(0.0)
                pa[0] = T.float32(0.0)
                pg[0] = T.float32(0.0)
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
                    vf = T.cast(v[i], "float32")
                    sig_v = T.sigmoid(T.cast(v0[i], "float32") + v12)
                    v_out[i] = T.cast(
                        vf + sig_v * (T.cast(v_first[i], "float32") - vf), DTYPE
                    )
                    w_out[i] = T.cast(
                        T.exp(-T.sigmoid(T.cast(w0[i], "float32") + w12) / T.float32(_SQRT_E)),
                        DTYPE,
                    )
                    a_val = T.sigmoid(T.cast(a0[i], "float32") + a12)
                    a_out[i] = T.cast(a_val, DTYPE)
                    kf = T.cast(k[i], "float32")
                    kk_out[i] = T.cast(kf * T.cast(k_k[i], "float32"), DTYPE)
                    k_out[i] = T.cast(
                        kf + T.cast(k_a[i], "float32") * (kf * a_val - kf), DTYPE
                    )
                    g_out[i] = T.cast(g12, DTYPE)

        return _impl

    @tilelang.jit(out_idx=[5])
    def _rank_gemv_kernel(C: int, Rv: int, Rw: int, Ra: int, Rg: int):
        @T.prim_func
        def _impl(
            xv: T.Tensor((C,), DTYPE),
            xw: T.Tensor((C,), DTYPE),
            xa: T.Tensor((C,), DTYPE),
            xg: T.Tensor((C,), DTYPE),
            W: T.Tensor((Rv + Rw + Ra + Rg, C), DTYPE),
            out: T.Tensor((Rv + Rw + Ra + Rg,), DTYPE),
        ):
            """Packed low-rank first-step GEMVs: one warp per rank row.

            Computes ``[v1t; w1t; a1t; g1t] @ [xv; xw; xa; xg]`` into one output
            vector, selecting the input by row segment. Replaces 4 cuBLAS fp16
            GEMVs (pathologically slow on Turing) with one hand-written GEMV.
            """
            with T.Kernel(Rv + Rw + Ra + Rg, threads=WARP) as (j,):
                n = T.get_thread_binding(0)
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = T.float32(0.0)
                for c in T.serial(C // WARP):
                    c_idx = n * (C // WARP) + c
                    x = T.if_then_else(
                        j < Rv,
                        xv[c_idx],
                        T.if_then_else(
                            j < Rv + Rw,
                            xw[c_idx],
                            T.if_then_else(j < Rv + Rw + Ra, xa[c_idx], xg[c_idx]),
                        ),
                    )
                    acc[0] += T.cast(W[j, c_idx], "float32") * T.cast(x, "float32")
                total = T.warp_reduce_sum(acc[0])
                if n == 0:
                    out[j] = T.cast(total, DTYPE)

        return _impl

    def fused_rank_gemv(
        xv: Tensor,
        xw: Tensor,
        xa: Tensor,
        xg: Tensor,
        packed: Tensor,
        rv: int,
        rw: int,
        ra: int,
        rg: int,
    ) -> Tensor:
        """Packed low-rank first-step GEMVs: ``[v1t; w1t; a1t; g1t] @ [xv; xw; xa; xg]``.

        Args:
            xv/xw/xa/xg: Shifted activations ``[C]``.
            packed: ``cat([v1t, w1t, a1t, g1t], dim=0)``, ``[rv+rw+ra+rg, C]``.
            rv/rw/ra/rg: Rank of each gate (RWKV7 checkpoints use distinct
                ranks, e.g. v=32 / w=64 / a=64 / g=128).

        Returns:
            Packed ``[rv+rw+ra+rg]`` result; slice per-gate.
        """
        if xv.device.type != "cuda":
            return torch.cat(
                [
                    packed[:rv] @ xv,
                    packed[rv : rv + rw] @ xw,
                    packed[rv + rw : rv + rw + ra] @ xa,
                    packed[rv + rw + ra :] @ xg,
                ]
            )
        return _rank_gemv_kernel(xv.shape[0], rv, rw, ra, rg)(xv, xw, xa, xg, packed)

    def fused_w_gate(x: Tensor, w0: Tensor) -> Tensor:
        """Fused w decay gate: w = exp(-sigmoid(w0 + x) / sqrt(e))."""
        if x.device.type != "cuda":
            return torch.exp(-torch.sigmoid(w0 + x) / _SQRT_E)
        return _w_gate_kernel(x.shape[0])(x, w0)

    def fused_v_gate(v: Tensor, v_first: Tensor, v0: Tensor, v12: Tensor) -> Tensor:
        """Fused v residual gate: v + sigmoid(v0 + v12) * (v_first - v)."""
        if v.device.type != "cuda":
            return v + torch.sigmoid(v0 + v12) * (v_first - v)
        return _v_gate_kernel(v.shape[0])(v, v_first, v0, v12)

    def fused_a_kk_k(
        a0: Tensor,
        a_x: Tensor,
        k: Tensor,
        k_k: Tensor,
        k_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Fused a-gate + kk + k LERP.

        Computes a = sigmoid(a0 + a_x), kk = k * k_k, new_k = k + k_a * (k * a - k).
        """
        if k.device.type != "cuda":
            a = torch.sigmoid(a0 + a_x)
            return a, k * k_k, k + k_a * (k * a - k)
        return _a_kk_k_kernel(k.shape[0])(a0, a_x, k, k_k, k_a)

    def fused_gates(
        vr: Tensor,
        wr: Tensor,
        ar: Tensor,
        gr: Tensor,
        v: Tensor,
        v_first: Tensor,
        k: Tensor,
        v2t: Tensor,
        w2t: Tensor,
        a2t: Tensor,
        g2t: Tensor,
        v0: Tensor,
        w0: Tensor,
        a0: Tensor,
        k_k: Tensor,
        k_a: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused low-rank gate second steps + v/w/a gate math.

        Args:
            vr/wr/ar/gr: Rank-out inputs ``[R]`` (first-step ``w1t @ x`` results).
            v2t/w2t/a2t/g2t: Rank weights transposed to ``[C, R]``.
            v/v_first/k: Gate state vectors ``[C]``.
            v0/w0/a0/k_k/k_a: Gate biases / scale vectors ``[C]``.

        Returns:
            ``(v, w, a, kk, k, g)``, each ``[C]``.
        """
        if v.device.type != "cuda":
            v12 = v2t @ vr
            w12 = w2t @ torch.tanh(wr)
            a12 = a2t @ ar
            g12 = g2t @ torch.sigmoid(gr)
            v_out = v + torch.sigmoid(v0 + v12) * (v_first - v)
            w_out = torch.exp(-torch.sigmoid(w0 + w12) / _SQRT_E)
            a = torch.sigmoid(a0 + a12)
            return v_out, w_out, a, k * k_k, k + k_a * (k * a - k), g12
        return _gates_kernel(
            vr.shape[0], wr.shape[0], ar.shape[0], gr.shape[0], v.shape[0]
        )(
            vr, wr, ar, gr, v2t, w2t, a2t, g2t, v, v_first, k, v0, w0, a0, k_k, k_a
        )

    return SimpleNamespace(
        fused_w_gate=fused_w_gate,
        fused_v_gate=fused_v_gate,
        fused_a_kk_k=fused_a_kk_k,
        fused_gates=fused_gates,
        fused_rank_gemv=fused_rank_gemv,
    )
