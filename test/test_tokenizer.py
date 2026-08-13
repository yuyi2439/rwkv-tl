"""Tokenizer tests against the vocab packaged with the library."""

from __future__ import annotations

from rwkv_tl import Tokenizer


def test_default_vocab_roundtrip() -> None:
    tok = Tokenizer()
    text = "Hello, 世界! How are you?"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(0 <= i < 65536 for i in ids)


def test_explicit_vocab_path() -> None:
    from importlib.resources import files

    from rwkv_tl.tokenizer import _DEFAULT_VOCAB

    vocab = files("rwkv_tl").joinpath(_DEFAULT_VOCAB)
    with vocab.open("r", encoding="utf-8"):
        pass  # packaged file is readable
    tok = Tokenizer(vocab)
    assert tok.decode(tok.encode("OK")) == "OK"
