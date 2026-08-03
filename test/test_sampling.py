"""Unit tests for the token sampling helpers (rwkv_tl.sampling)."""

from __future__ import annotations

import pytest
import torch

from rwkv_tl.sampling import sample_logits


def test_greedy_argmax() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5, 3.0])
    assert sample_logits(logits) == 3
    assert sample_logits(logits, temperature=0.0) == 3
    assert sample_logits(logits, temperature=-1.0) == 3


def test_top_k_one_is_greedy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(100)
    for _ in range(20):
        assert sample_logits(logits, temperature=1.0, top_k=1) == int(logits.argmax())


def test_repetition_penalty_flips_near_tie() -> None:
    # top_k=1 picks the max logit deterministically (penalized when seen).
    logits = torch.tensor([0.5, 0.4])
    assert sample_logits(logits, temperature=1.0, top_k=1) == 0
    # penalizing token 0 (0.5 / 2 = 0.25) makes token 1 the max.
    assert (
        sample_logits(
            logits, temperature=1.0, top_k=1, repetition_penalty=2.0, seen=[0]
        )
        == 1
    )


def test_sample_in_range() -> None:
    torch.manual_seed(0)
    logits = torch.randn(100)
    for _ in range(20):
        t = sample_logits(logits, temperature=1.0, top_k=50, top_p=0.9)
        assert 0 <= t < 100


@pytest.mark.parametrize("temperature", [None, 0.0, 1.0])
def test_generate_shape_and_dtype(temperature) -> None:
    # sample_logits never returns an out-of-range id regardless of settings.
    torch.manual_seed(1)
    logits = torch.randn(65536)
    t = sample_logits(logits, temperature=temperature)
    assert isinstance(t, int)
    assert 0 <= t < 65536
