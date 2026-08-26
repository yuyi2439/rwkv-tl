"""User-facing text API through the pure-torch backend (CPU-friendly).

Requires ``RWKV_CHECKPOINT_PATH``; runs on CPU when CUDA is unavailable, so
the text-level wiring (tokenizer + model composition) can be validated
without a GPU. The tilelang path needs CUDA and is validated on GPU machines
by the other tests.
"""

from __future__ import annotations

import os

import pytest

import rwkv_tl
from rwkv_tl.core import RWKV7Weight

CKPT = os.environ.get("RWKV_CHECKPOINT_PATH")
if not CKPT:
    pytest.skip("RWKV_CHECKPOINT_PATH not set", allow_module_level=True)


@pytest.fixture(scope="module")
def model() -> rwkv_tl.RWKV7TextModel:
    assert CKPT is not None
    w = RWKV7Weight(CKPT, device="cpu")
    return rwkv_tl.rwkv7(
        w,
        backend="torch",
        use_graph=False,
        is_torch_compile=False,
    )


def test_encode_detokenize(model) -> None:
    text = "The meaning of life"
    assert model.detokenize(model.tokenize(text)) == text
