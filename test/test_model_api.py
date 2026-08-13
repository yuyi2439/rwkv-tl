"""User-facing text API through the pure-torch backend (CPU-friendly).

Requires ``RWKV_CHECKPOINT_PATH``; runs on CPU when CUDA is unavailable, so
the whole text-level surface (tune_state / logits / generate / state I/O) can
be validated without a GPU. The tilelang path needs CUDA and is validated on
GPU machines by the other tests.
"""

from __future__ import annotations

import os

import pytest
import torch

import rwkv_tl

CKPT = os.environ.get("RWKV_CHECKPOINT_PATH")
if not CKPT:
    pytest.skip("RWKV_CHECKPOINT_PATH not set", allow_module_level=True)


@pytest.fixture(scope="module")
def model() -> rwkv_tl.RWKV7Torch:
    return rwkv_tl.rwkv7(
        CKPT,
        device="cpu",
        backend="torch",
        use_graph=False,
        is_torch_compile=False,
    )


def test_encode_detokenize(model) -> None:
    text = "The meaning of life"
    assert model.detokenize(model.tokenize(text)) == text


def test_tune_state_and_logits(model) -> None:
    S = model.tune_state("The meaning of life is")
    logits, _ = model.logits(" to", state=S)
    assert tuple(logits.shape) == (65536,)
    assert torch.isfinite(logits.float()).all()


def test_generate_text_and_tokens(model) -> None:
    out = model.generate("The meaning of life is", max_tokens=2)
    assert isinstance(out, str) and len(out) > 0
    ids = model.generate(model.tokenize("The meaning of life is"), max_tokens=2)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)


def test_state_tune_save_load_attach(tmp_path, model) -> None:
    S = model.tune_state("You are a helpful assistant.")
    path = tmp_path / "tuned.pt"
    S.save(path)
    loaded = rwkv_tl.core.State.load(path)
    out = model.generate("Hi", state=loaded, max_tokens=2)
    assert isinstance(out, str)
