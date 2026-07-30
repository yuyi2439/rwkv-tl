"""Fused elementwise kernels built with tilelang.

Each kernel merges multiple elementwise ops into one CUDA launch to remove
intermediate tensors and launch overhead. Kernels are compiled once at module
load and cached; outputs are returned as torch tensors via out_idx.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from torch import Tensor

# RWKV7-0.1B constants
N_EMBD = 768
BLOCK = 256
GRID = (N_EMBD + BLOCK - 1) // BLOCK


@T.prim_func
def _fused_lerp6(
    x: T.Tensor((768,), "bfloat16"),
    prev: T.Tensor((768,), "bfloat16"),
    x_r: T.Tensor((768,), "bfloat16"),
    x_w: T.Tensor((768,), "bfloat16"),
    x_k: T.Tensor((768,), "bfloat16"),
    x_v: T.Tensor((768,), "bfloat16"),
    x_a: T.Tensor((768,), "bfloat16"),
    x_g: T.Tensor((768,), "bfloat16"),
    xr: T.Tensor((768,), "bfloat16"),
    xw: T.Tensor((768,), "bfloat16"),
    xk: T.Tensor((768,), "bfloat16"),
    xv: T.Tensor((768,), "bfloat16"),
    xa: T.Tensor((768,), "bfloat16"),
    xg: T.Tensor((768,), "bfloat16"),
):
    """Fused 6x LERP: out_i = x + x_i_weight * (prev - x).

    Replaces 18 elementwise kernels and 18 intermediate tensors from the torch
    form with a single kernel and zero intermediates. Grid: 3 blocks x 256
    threads = 768 threads, one element per thread.
    """
    for bx in T.thread_binding(GRID, "blockIdx.x"):
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < N_EMBD:
                xi = x[i]
                pi = prev[i]
                diff = pi - xi
                xr[i] = xi + x_r[i] * diff
                xw[i] = xi + x_w[i] * diff
                xk[i] = xi + x_k[i] * diff
                xv[i] = xi + x_v[i] * diff
                xa[i] = xi + x_a[i] * diff
                xg[i] = xi + x_g[i] * diff


# Compile once at module load; out_idx marks tensors 8-13 as outputs.
_fused_lerp6_kernel = tilelang.compile(_fused_lerp6, out_idx=[8, 9, 10, 11, 12, 13])


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
    """Fused LERP(x, prev, weight) for six weight/output pairs.

    Args:
        x: Current token hidden vector after LayerNorm, [N_EMBD].
        prev: Previous token's x, [N_EMBD].
        x_r/x_w/x_k/x_v/x_a/x_g: Six LERP weights, each [N_EMBD].

    Returns:
        (xr, xw, xk, xv, xa, xg), each [N_EMBD], bf16.
    """
    return _fused_lerp6_kernel(x, prev, x_r, x_w, x_k, x_v, x_a, x_g)
