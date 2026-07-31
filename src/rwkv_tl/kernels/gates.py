"""Fused gate kernels (elementwise over the embedding dim C).

All gate math (sigmoid / exp / LERP) runs in fp32 and is cast to bf16 only at
the store, favouring throughput over bit-exactness with PyTorch eager.
"""
from __future__ import annotations

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

from ._common import _SQRT_E, BLOCK

C = T.dynamic("C")


@T.prim_func
def _fused_w_gate(
    x: T.Tensor((C,), "bfloat16"),
    w0: T.Tensor((C,), "bfloat16"),
    out: T.Tensor((C,), "bfloat16"),
):
    """Fused w decay gate: w = exp(-sigmoid(w0 + x) / sqrt(e))."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < C:
                s = T.cast(x[i], "float32") + T.cast(w0[i], "float32")
                out[i] = T.cast(T.exp(-T.sigmoid(s) / T.float32(_SQRT_E)), "bfloat16")


_fused_w_gate_kernel = tilelang.compile(_fused_w_gate, out_idx=[2])


@T.prim_func
def _fused_v_gate(
    v: T.Tensor((C,), "bfloat16"),
    v_first: T.Tensor((C,), "bfloat16"),
    v0: T.Tensor((C,), "bfloat16"),
    v12: T.Tensor((C,), "bfloat16"),
    out: T.Tensor((C,), "bfloat16"),
):
    """Fused v residual gate: v + sigmoid(v0 + v12) * (v_first - v)."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < C:
                vf = T.cast(v[i], "float32")
                sig = T.sigmoid(T.cast(v0[i], "float32") + T.cast(v12[i], "float32"))
                out[i] = T.cast(vf + sig * (T.cast(v_first[i], "float32") - vf), "bfloat16")


_fused_v_gate_kernel = tilelang.compile(_fused_v_gate, out_idx=[4])


@T.prim_func
def _fused_a_kk_k(
    a0: T.Tensor((C,), "bfloat16"),
    a_x: T.Tensor((C,), "bfloat16"),
    k: T.Tensor((C,), "bfloat16"),
    k_k: T.Tensor((C,), "bfloat16"),
    k_a: T.Tensor((C,), "bfloat16"),
    a_out: T.Tensor((C,), "bfloat16"),
    kk_out: T.Tensor((C,), "bfloat16"),
    k_out: T.Tensor((C,), "bfloat16"),
):
    """Fused a-gate + kk + k LERP.

    a = sigmoid(a0 + a_x); kk = k * k_k; new_k = k + k_a * (k * a - k).
    """
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < C:
                kf = T.cast(k[i], "float32")
                a_val = T.sigmoid(T.cast(a0[i], "float32") + T.cast(a_x[i], "float32"))
                a_out[i] = T.cast(a_val, "bfloat16")
                kk_out[i] = T.cast(kf * T.cast(k_k[i], "float32"), "bfloat16")
                k_out[i] = T.cast(kf + T.cast(k_a[i], "float32") * (kf * a_val - kf), "bfloat16")


_fused_a_kk_k_kernel = tilelang.compile(_fused_a_kk_k, out_idx=[5, 6, 7])


@T.prim_func
def _fused_neg_kk_a(
    kk: T.Tensor((C,), "bfloat16"),
    a: T.Tensor((C,), "bfloat16"),
    b: T.Tensor((C,), "bfloat16"),
):
    """Fused negation + multiply: B = -kk * a."""
    for bx in T.thread_binding((C + BLOCK - 1) // BLOCK, "blockIdx.x"):
        for tx in T.thread_binding(BLOCK, "threadIdx.x"):
            i = bx * BLOCK + tx
            if i < C:
                b[i] = T.cast(-(T.cast(kk[i], "float32") * T.cast(a[i], "float32")), "bfloat16")


_fused_neg_kk_a_kernel = tilelang.compile(_fused_neg_kk_a, out_idx=[2])


# --------------------------------------------------------------------------- #
#  Python wrappers with CPU fallback
# --------------------------------------------------------------------------- #
def fused_w_gate(x: Tensor, w0: Tensor) -> Tensor:
    """Fused w decay gate: w = exp(-sigmoid(w0 + x) / sqrt(e)).

    Args:
        x: Matmul result (tanh(xw @ w1) @ w2), [C].
        w0: Bias vector, [C].

    Returns:
        w: Decay gate, [C], bf16.
    """
    if x.device.type != "cuda":
        return torch.exp(-torch.sigmoid(w0 + x) / _SQRT_E)
    return _fused_w_gate_kernel(x, w0)


def fused_v_gate(v: Tensor, v_first: Tensor, v0: Tensor, v12: Tensor) -> Tensor:
    """Fused v residual gate: v + sigmoid(v0 + v12) * (v_first - v).

    Args:
        v: Current value, [C].
        v_first: First-layer value (residual), [C].
        v0: Bias vector, [C].
        v12: Matmul result (xv @ v1 @ v2), [C].

    Returns:
        Gated value, [C], bf16.
    """
    if v.device.type != "cuda":
        return v + torch.sigmoid(v0 + v12) * (v_first - v)
    return _fused_v_gate_kernel(v, v_first, v0, v12)


def fused_a_kk_k(
    a0: Tensor, a_x: Tensor, k: Tensor, k_k: Tensor, k_a: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused a-gate + kk + k LERP.

    Computes a = sigmoid(a0 + a_x), kk = k * k_k, new_k = k + k_a * (k * a - k).

    Args:
        a0: Bias for a gate, [C].
        a_x: Matmul result (xa @ a1 @ a2), [C].
        k: Key vector, [C].
        k_k: kk scaling factor, [C].
        k_a: k LERP weight, [C].

    Returns:
        (a, kk, new_k), each [C], bf16.
    """
    if k.device.type != "cuda":
        a = torch.sigmoid(a0 + a_x)
        return a, k * k_k, k + k_a * (k * a - k)
    return _fused_a_kk_k_kernel(a0, a_x, k, k_k, k_a)


def fused_neg_kk_a(kk: Tensor, a: Tensor) -> Tensor:
    """Fused negation + multiply: B = -kk * a.

    Args:
        kk: Normalized kk, [C].
        a: Activation gate, [C].

    Returns:
        B: -kk * a, [C], bf16.
    """
    if kk.device.type != "cuda":
        return -kk * a
    return _fused_neg_kk_a_kernel(kk, a)
