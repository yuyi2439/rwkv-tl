"""State save/load round-trip (CPU tensors, no checkpoint needed)."""

from __future__ import annotations

import pytest
import torch

from rwkv_tl.state import State


def test_save_load_roundtrip(tmp_path) -> None:
    s = State(2, 768, 64, device="cpu", dtype=torch.float16)
    s.tmix[0]["x"][:3] = 1.5
    s.tmix[1]["rnn"][0, 0, :2] = 0.25
    s.cmix[1]["x"][5] = -2.0

    path = tmp_path / "state.pt"
    s.save(path)
    loaded = State.load(path)

    assert isinstance(loaded, State)
    assert len(loaded.tmix) == 2 and len(loaded.cmix) == 2
    assert torch.equal(loaded.tmix[0]["x"], s.tmix[0]["x"])
    assert torch.equal(loaded.tmix[1]["rnn"], s.tmix[1]["rnn"])
    assert torch.equal(loaded.cmix[1]["x"], s.cmix[1]["x"])


def test_load_dtype_conversion_keeps_rnn_fp32(tmp_path) -> None:
    s = State(1, 64, 64, device="cpu", dtype=torch.float16)
    path = tmp_path / "state.pt"
    s.save(path)

    loaded = State.load(path, dtype=torch.bfloat16)
    assert loaded.tmix[0]["x"].dtype == torch.bfloat16
    assert loaded.cmix[0]["x"].dtype == torch.bfloat16
    assert loaded.tmix[0]["rnn"].dtype == torch.float32


def test_load_rejects_foreign_file(tmp_path) -> None:
    path = tmp_path / "not_a_state.pt"
    torch.save({"foo": 1}, path)
    with pytest.raises(ValueError, match="not a saved rwkv_tl State"):
        State.load(path)
