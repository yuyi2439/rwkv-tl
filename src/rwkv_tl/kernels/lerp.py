"""Fused LERP kernels (elementwise over the embedding dim C).

Each kernel merges the x + w * (prev - x) LERP chain into one launch. The
*_copy variants additionally write x into a caller-supplied buffer (state["x"])
in-place, eliminating a separate copy_ call.
"""

# tilelang's @T.prim_func DSL uses call expressions (T.Tensor(...)) in type
# positions and tilelang-only intrinsics; pyright cannot type-check those.
# pyright: reportInvalidTypeForm=false, reportCallIssue=false, reportAttributeAccessIssue=false
from __future__ import annotations

import tilelang
import tilelang.language as T
from torch import Tensor

from ._common import BLOCK
from .gemm import fused_rkv_gemm

C = T.dynamic("C")


@T.prim_func
def _fused_lerp6(
    x: T.Tensor((C,), "bfloat16"),
    prev: T.Tensor((C,), "bfloat16"),
    x_r: T.Tensor((C,), "bfloat16"),
    x_w: T.Tensor((C,), "bfloat16"),
    x_k: T.Tensor((C,), "bfloat16"),
    x_v: T.Tensor((C,), "bfloat16"),
    x_a: T.Tensor((C,), "bfloat16"),
    x_g: T.Tensor((C,), "bfloat16"),
    xr: T.Tensor((C,), "bfloat16"),
    xw: T.Tensor((C,), "bfloat16"),
    xk: T.Tensor((C,), "bfloat16"),
    xv: T.Tensor((C,), "bfloat16"),
    xa: T.Tensor((C,), "bfloat16"),
    xg: T.Tensor((C,), "bfloat16"),
):
    """Fused 6x LERP: out_i = x + w_i * (prev - x)."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):  # type: ignore[operator]
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
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


_fused_lerp6_kernel = tilelang.compile(_fused_lerp6, out_idx=[8, 9, 10, 11, 12, 13])


@T.prim_func
def _fused_lerp6_copy(
    x: T.Tensor((C,), "bfloat16"),
    prev: T.Tensor((C,), "bfloat16"),
    x_r: T.Tensor((C,), "bfloat16"),
    x_w: T.Tensor((C,), "bfloat16"),
    x_k: T.Tensor((C,), "bfloat16"),
    x_v: T.Tensor((C,), "bfloat16"),
    x_a: T.Tensor((C,), "bfloat16"),
    x_g: T.Tensor((C,), "bfloat16"),
    x_copy: T.Tensor((C,), "bfloat16"),
    xr: T.Tensor((C,), "bfloat16"),
    xw: T.Tensor((C,), "bfloat16"),
    xk: T.Tensor((C,), "bfloat16"),
    xv: T.Tensor((C,), "bfloat16"),
    xa: T.Tensor((C,), "bfloat16"),
    xg: T.Tensor((C,), "bfloat16"),
):
    """Fused 6x LERP + copy x to x_copy buffer (in-place)."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):  # type: ignore[operator]
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
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


_fused_lerp6_copy_kernel = tilelang.compile(
    _fused_lerp6_copy, out_idx=[9, 10, 11, 12, 13, 14]
)


@T.prim_func
def _fused_lerp1_copy(
    x: T.Tensor((C,), "bfloat16"),
    prev: T.Tensor((C,), "bfloat16"),
    w: T.Tensor((C,), "bfloat16"),
    x_copy: T.Tensor((C,), "bfloat16"),
    out: T.Tensor((C,), "bfloat16"),
):
    """Fused single LERP + copy x to x_copy buffer (in-place)."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):  # type: ignore[operator]
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < C:
                x_copy[i] = x[i]
                out[i] = x[i] + w[i] * (prev[i] - x[i])


_fused_lerp1_copy_kernel = tilelang.compile(_fused_lerp1_copy, out_idx=[4])


# --------------------------------------------------------------------------- #
#  Python wrappers with CPU fallback
# --------------------------------------------------------------------------- #
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
    """Fused 6x LERP: x + w_i * (prev - x) for six weight/output pairs.

    Args:
        x: Current hidden vector after LayerNorm, [C].
        prev: previous x, [C].
        x_r/x_w/x_k/x_v/x_a/x_g: six LERP weights, each [C].

    Returns:
        (xr, xw, xk, xv, xa, xg), each [C], bf16.
    """
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
    return _fused_lerp6_kernel(x, prev, x_r, x_w, x_k, x_v, x_a, x_g)


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
    """Fused 6x LERP + copy x to x_copy buffer (in-place).

    Writes x into the caller-supplied x_copy buffer (state["x"]) in-place,
    eliminating a separate state["x"].copy_(x) call.

    Args:
        x: Current hidden vector after LayerNorm, [C].
        prev: previous x, [C].
        x_r/x_w/x_k/x_v/x_a/x_g: six LERP weights, each [C].
        x_copy: caller-supplied buffer (state["x"]) written in-place, [C].

    Returns:
        (xr, xw, xk, xv, xa, xg), each [C], bf16.
    """
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
    return _fused_lerp6_copy_kernel(x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy)


def fused_lerp1_copy(x: Tensor, prev: Tensor, w: Tensor, x_copy: Tensor) -> Tensor:
    """Fused single LERP + copy x to x_copy buffer (in-place).

    Args:
        x: Current value, [C].
        prev: previous value, [C].
        w: LERP weight, [C].
        x_copy: caller-supplied buffer (state["x"]) written in-place, [C].

    Returns:
        Interpolated value, [C], bf16.
    """
    if x.device.type != "cuda":
        x_copy.copy_(x)
        return x + w * (prev - x)
    return _fused_lerp1_copy_kernel(x, prev, w, x_copy)


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
    """Run lerp6(+copy) and r/k/v projections in one Python call.

    This combines:
    1) fused_lerp6_copy(x, prev, ...)
    2) fused_rkv_gemm on (xr, xk, xv)

    It reduces decode-side Python dispatch and replaces three separate mv calls
    with one batched GEMM path.

    Returns:
        (r, k, v, xv, xw, xa, xg), each [C].
    """
    xr, xw, xk, xv, xa, xg = fused_lerp6_copy(
        x, prev, x_r, x_w, x_k, x_v, x_a, x_g, x_copy
    )
    rkv = fused_rkv_gemm(
        xr.unsqueeze(0),
        xk.unsqueeze(0),
        xv.unsqueeze(0),
        rWt_stack,
    )
    return rkv[0, 0], rkv[1, 0], rkv[2, 0], xv, xw, xa, xg
