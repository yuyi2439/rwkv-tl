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

from ._common import BLOCK

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

    return SimpleNamespace(
        fused_w_gate=fused_w_gate,
        fused_v_gate=fused_v_gate,
        fused_a_kk_k=fused_a_kk_k,
    )
