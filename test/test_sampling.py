"""Unit tests for the token sampling helpers (rwkv_tl.sampling)."""

from __future__ import annotations

import pytest
import torch

from rwkv_tl.sampling import apply_stop, sample_logits


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


def test_apply_stop_matches_and_truncates() -> None:
    out = [1, 2, 3, 4, 5, 6]
    assert apply_stop(out, [[7, 8]]) is False
    assert out == [1, 2, 3, 4, 5, 6]
    assert apply_stop(out, [[9, 10], [5, 6]]) is True
    assert out == [1, 2, 3, 4]


def test_apply_stop_longer_sequence() -> None:
    # A stop sequence longer than the generated tail never matches.
    out = [1, 2, 3]
    assert apply_stop(out, [[1, 2, 3, 4]]) is False
    assert out == [1, 2, 3]


def test_apply_stop_disabled_and_empty() -> None:
    out = [1, 2, 3]
    assert apply_stop(out, None) is False
    assert apply_stop(out, []) is False
    assert apply_stop(out, [[]]) is False
    assert out == [1, 2, 3]


def test_apply_stop_first_match_wins() -> None:
    out = [1, 2, 3, 4]
    assert apply_stop(out, [[3, 4], [2, 3, 4]]) is True
    assert out == [1, 2]
