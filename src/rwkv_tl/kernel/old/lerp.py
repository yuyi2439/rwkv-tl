"""Fused LERP kernels (elementwise over the embedding dim C), dtype-parameterized.

Each kernel merges the x + w * (prev - x) LERP chain into one launch. The
*_copy variants additionally write x into a caller-supplied buffer (state["x"])
in-place, eliminating a separate copy_ call. C is a model constant baked at
compile time (compiled per-C, cached); only per-call sizes stay dynamic.

``build(DTYPE, fused_rkv_gemm)`` returns a namespace bound to one element type;
``fused_lerp6_rkv_copy`` needs the matching dtype-bound GEMM from ``gemm.build``.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
# NOTE: no `from __future__ import annotations` here -- tilelang's eager builder
# evaluates the annotation expressions, and a stringified annotation would lose
# the closure DTYPE param (NameError).

from collections.abc import Callable
from types import SimpleNamespace

import tilelang
import tilelang.language as T
from torch import Tensor

from ._common import BLOCK


def build(DTYPE: str, fused_rkv_gemm: Callable) -> SimpleNamespace:
    """Build the dtype-bound LERP kernel namespace.

    Args:
        DTYPE: Element type string, ``"float16"`` or ``"bfloat16"``.
        fused_rkv_gemm: The dtype-matched GEMM from ``gemm.build(DTYPE)``,
            used by ``fused_lerp6_rkv_copy``.

    Returns:
        ``SimpleNamespace`` with ``fused_lerp6``, ``fused_lerp6_copy``,
        ``fused_lerp1_copy``, ``fused_lerp6_rkv_copy``.
    """

    @tilelang.jit(out_idx=[8, 9, 10, 11, 12, 13])
    def _lerp6_kernel(C: int):
        @T.prim_func
        def _impl(
            x: T.Tensor((C,), DTYPE),
            prev: T.Tensor((C,), DTYPE),
            x_r: T.Tensor((C,), DTYPE),
            x_w: T.Tensor((C,), DTYPE),
            x_k: T.Tensor((C,), DTYPE),
            x_v: T.Tensor((C,), DTYPE),
            x_a: T.Tensor((C,), DTYPE),
            x_g: T.Tensor((C,), DTYPE),
            xr: T.Tensor((C,), DTYPE),
            xw: T.Tensor((C,), DTYPE),
            xk: T.Tensor((C,), DTYPE),
            xv: T.Tensor((C,), DTYPE),
            xa: T.Tensor((C,), DTYPE),
            xg: T.Tensor((C,), DTYPE),
        ):
            """Fused 6x LERP: out_i = x + w_i * (prev - x)."""
            with T.Kernel(T.ceildiv(C, BLOCK), threads=BLOCK) as bx:
                tx = T.get_thread_binding(0)
                i = bx * BLOCK + tx
                if i < C:
                    xi = x[i]
                    diff = prev[i] - xi
                    xr[i] = xi + x_r[i] * diff
                    xw[i] = xi + x_w[i] * diff
                    xk[i] = xi + x_k[i] * diff
                    xv[i] = xi + x_v[i] * diff
                    xa[i] = xi + x_a[i] * diff
                    xg[i] = xi + x_g[i] * diff

        return _impl

    @tilelang.jit(out_idx=[9, 10, 11, 12, 13, 14])
    def _lerp6_copy_kernel(C: int):
        @T.prim_func
        def _impl(
            x: T.Tensor((C,), DTYPE),
            prev: T.Tensor((C,), DTYPE),
            x_r: T.Tensor((C,), DTYPE),
            x_w: T.Tensor((C,), DTYPE),
            x_k: T.Tensor((C,), DTYPE),
            x_v: T.Tensor((C,), DTYPE),
            x_a: T.Tensor((C,), DTYPE),
            x_g: T.Tensor((C,), DTYPE),
            x_copy: T.Tensor((C,), DTYPE),
            xr: T.Tensor((C,), DTYPE),
            xw: T.Tensor((C,), DTYPE),
            xk: T.Tensor((C,), DTYPE),
            xv: T.Tensor((C,), DTYPE),
            xa: T.Tensor((C,), DTYPE),
            xg: T.Tensor((C,), DTYPE),
        ):
            """Fused 6x LERP + copy x to x_copy buffer (in-place)."""
            with T.Kernel(T.ceildiv(C, BLOCK), threads=BLOCK) as bx:
                tx = T.get_thread_binding(0)
                i = bx * BLOCK + tx
                if i < C:
                    xi = x[i]
                    diff = prev[i] - xi
                    x_copy[i] = xi
                    xr[i] = xi + x_r[i] * diff
                    xw[i] = xi + x_w[i] * diff
                    xk[i] = xi + x_k[i] * diff
                    xv[i] = xi + x_v[i] * diff
                    xa[i] = xi + x_a[i] * diff
                    xg[i] = xi + x_g[i] * diff

        return _impl

    @tilelang.jit(out_idx=[4])
    def _lerp1_copy_kernel(C: int):
        @T.prim_func
        def _impl(
            x: T.Tensor((C,), DTYPE),
            prev: T.Tensor((C,), DTYPE),
            w: T.Tensor((C,), DTYPE),
            x_copy: T.Tensor((C,), DTYPE),
            out: T.Tensor((C,), DTYPE),
        ):
            """Fused single LERP + copy x to x_copy buffer (in-place)."""
            with T.Kernel(T.ceildiv(C, BLOCK), threads=BLOCK) as bx:
                for i in T.Parallel(BLOCK):
                    idx = bx * BLOCK + i
                    if idx < C:
                        x_copy[idx] = x[idx]
                        out[idx] = x[idx] + w[idx] * (prev[idx] - x[idx])

        return _impl

    def fused_lerp6(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused 6x LERP: x + w_i * (prev - x) for six weight/output pairs."""
        if x.device.type != "cuda":
            diff = prev - x
            return (
                x + x_r * diff,
                x + x_w * diff,
                x + x_k * diff,
                x + x_v * diff,
                x + x_a * diff,
                x + x_g * diff,
            )
        return _lerp6_kernel(x.shape[0])(x, prev, x_r, x_w, x_k, x_v, x_a, x_g)

    def fused_lerp6_copy(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused 6x LERP + copy x to x_copy buffer (in-place)."""
        if x.device.type != "cuda":
            diff = prev - x
            x_copy.copy_(x)
            return (
                x + x_r * diff,
                x + x_w * diff,
                x + x_k * diff,
                x + x_v * diff,
                x + x_a * diff,
                x + x_g * diff,
            )
        return _lerp6_copy_kernel(x.shape[0])(
            x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy
        )

    def fused_lerp1_copy(x: Tensor, prev: Tensor, w: Tensor, x_copy: Tensor) -> Tensor:
        """Fused single LERP + copy x to x_copy buffer (in-place)."""
        if x.device.type != "cuda":
            # Compute the lerp first: prev aliases x_copy (both state["x"]), so an
            # earlier copy_ would corrupt prev before it is read.
            out = x + w * (prev - x)
            x_copy.copy_(x)
            return out
        return _lerp1_copy_kernel(x.shape[0])(x, prev, w, x_copy)

    def fused_lerp6_rkv_copy(
        x: Tensor,
        prev: Tensor,
        x_r: Tensor,
        x_w: Tensor,
        x_k: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        x_g: Tensor,
        x_copy: Tensor,
        rWt_stack: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run lerp6(+copy) and r/k/v projections in one Python call."""
        xr, xw, xk, xv, xa, xg = fused_lerp6_copy(
            x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy
        )
        rkv = fused_rkv_gemm(
            xr.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0), rWt_stack
        )
        return rkv[0, 0], rkv[1, 0], rkv[2, 0], xv, xw, xa, xg

    return SimpleNamespace(
        fused_lerp6=fused_lerp6,
        fused_lerp6_copy=fused_lerp6_copy,
        fused_lerp1_copy=fused_lerp1_copy,
        fused_lerp6_rkv_copy=fused_lerp6_rkv_copy,
    )
