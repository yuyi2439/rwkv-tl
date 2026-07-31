"""Correctness of GraphDecoder against baseline forward.

Verifies that the CUDA Graph-accelerated single-token decoder produces
bit-identical logits to the plain ``model.forward`` path over a multi-token
sequence.
"""
from __future__ import annotations

import pytest
import torch

from rwkv_tl import RWKV7
from rwkv_tl.graph_decode import GraphDecoder

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]


@pytest.fixture(scope="module")
def model(ckpt_path: str, vocab_path: str) -> RWKV7:
    with torch.device("cuda"):
        return RWKV7(ckpt_path, vocab_path)


def _baseline_logits(model: RWKV7, tokens: list[int]) -> torch.Tensor:
    with torch.device("cuda"):
        S = model.zero_state()
        logits, _ = model.forward(tokens, S)
    return logits.float().cpu()


def _graph_logits(model: RWKV7, tokens: list[int]) -> torch.Tensor:
    dec = GraphDecoder(model)
    dec.reset()
    out = None
    for t in tokens:
        out = dec.step(t)
    assert out is not None
    return out.float().cpu()


def test_graph_logits_bit_exact(model: RWKV7) -> None:
    """GraphDecoder logits must be bit-identical to baseline forward."""
    ref = _baseline_logits(model, TOKENS)
    got = _graph_logits(model, TOKENS)
    diff = (ref - got).abs()
    assert diff.max().item() == 0.0, f"max_abs={diff.max().item()}"


def test_graph_argmax_match(model: RWKV7) -> None:
    """Argmax of final logits must match between graph and baseline."""
    ref = _baseline_logits(model, TOKENS)
    got = _graph_logits(model, TOKENS)
    assert int(ref.argmax()) == int(got.argmax())
