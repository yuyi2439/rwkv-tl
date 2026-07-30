"""End-to-end forward numerical consistency: tilelang vs pure-torch LERP.

Runs the full RWKV7 forward over a fixed 32-token sequence twice:
  1. with the integrated tilelang fused_lerp6
  2. with fused_lerp6 monkey-patched to a pure-torch 6x LERP reference

Acceptance: bit-exact logits (max_abs == 0), matching argmax and top-5 set.
"""
from __future__ import annotations

import pytest
import torch

import rwkv_tl
from rwkv_tl import RWKV7

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]


def _lerp6_ref(x, prev, x_r, x_w, x_k, x_v, x_a, x_g):
    """Pure-torch 6x LERP, signature-matched to fused_lerp6."""
    diff = prev - x
    return (
        x + x_r * diff,
        x + x_w * diff,
        x + x_k * diff,
        x + x_v * diff,
        x + x_a * diff,
        x + x_g * diff,
    )


def _run_forward(model: RWKV7, tokens: list[int]) -> torch.Tensor:
    # zero_state must run under the cuda device context so state tensors land on
    # GPU; tilelang kernels require cuda inputs.
    with torch.device("cuda"):
        S = model.zero_state()
        logits, _ = model.forward(tokens, S)
    return logits.float().cpu()


@pytest.fixture(scope="module")
def model(ckpt_path: str, vocab_path: str) -> RWKV7:
    with torch.device("cuda"):
        return RWKV7(ckpt_path, vocab_path)


def test_forward_logits_bit_exact(model: RWKV7) -> None:
    """Forward logits must be bit-identical between tilelang and torch LERP."""
    logits_tl = _run_forward(model, TOKENS)

    orig = rwkv_tl.fused_lerp6
    rwkv_tl.fused_lerp6 = _lerp6_ref
    try:
        logits_ref = _run_forward(model, TOKENS)
    finally:
        rwkv_tl.fused_lerp6 = orig

    diff = (logits_tl - logits_ref).abs()
    assert diff.max().item() == 0.0, f"max_abs={diff.max().item()}"
    assert diff.mean().item() == 0.0, f"mean_abs={diff.mean().item()}"


def test_forward_argmax_match(model: RWKV7) -> None:
    """Argmax of final logits must match between the two implementations."""
    logits_tl = _run_forward(model, TOKENS)

    orig = rwkv_tl.fused_lerp6
    rwkv_tl.fused_lerp6 = _lerp6_ref
    try:
        logits_ref = _run_forward(model, TOKENS)
    finally:
        rwkv_tl.fused_lerp6 = orig

    assert int(logits_tl.argmax()) == int(logits_ref.argmax())


def test_forward_top5_match(model: RWKV7) -> None:
    """Top-5 token set of final logits must match between implementations."""
    logits_tl = _run_forward(model, TOKENS)

    orig = rwkv_tl.fused_lerp6
    rwkv_tl.fused_lerp6 = _lerp6_ref
    try:
        logits_ref = _run_forward(model, TOKENS)
    finally:
        rwkv_tl.fused_lerp6 = orig

    top5_tl = set(torch.topk(logits_tl.reshape(-1), 5).indices.tolist())
    top5_ref = set(torch.topk(logits_ref.reshape(-1), 5).indices.tolist())
    assert top5_tl == top5_ref
