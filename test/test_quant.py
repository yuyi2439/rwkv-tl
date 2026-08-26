"""QTensor W8A16 quantization unit tests."""

import pytest
import torch

from rwkv_tl.quant import QTensor, quant_error, quantize, stack_qtensors


def test_quantize_dequant_roundtrip_shape_and_dtype() -> None:
    w = torch.randn(768, 768)
    qt = quantize(w, group=128)
    assert isinstance(qt, QTensor)
    assert qt.q.dtype == torch.int8
    assert qt.s.dtype == torch.float16
    assert qt.shape == (768, 768)
    assert qt.s.shape == (6, 768)
    d = qt.dequant()
    assert d.shape == w.shape and d.dtype == torch.float16


def test_quant_error_is_small() -> None:
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    qt = quantize(w, group=128)
    err = quant_error(w, qt)
    assert err["rel"] < 0.02
    assert err["cos"] > 0.999


def test_batched_layout() -> None:
    w = torch.randn(3, 768, 768)
    qt = quantize(w, group=128)
    assert qt.q.shape == (3, 768, 768)
    assert qt.s.shape == (3, 6, 768)
    assert qt.dequant().shape == w.shape


def test_stack_qtensors() -> None:
    w0 = torch.randn(768, 768)
    w1 = torch.randn(768, 768)
    st = stack_qtensors([quantize(w0), quantize(w1)])
    assert st.q.shape == (2, 768, 768)
    assert st.s.shape == (2, 6, 768)
    per = quant_error(torch.stack([w0, w1]), st)
    assert per["rel"] < 0.02


def test_group_must_divide_k() -> None:
    with pytest.raises(ValueError):
        quantize(torch.randn(100, 64), group=128)
