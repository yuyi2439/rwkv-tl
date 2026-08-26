"""Weight-bound kernel API: argument marshaling and factory param contracts.

The tilelang kernels themselves need CUDA, so these tests exercise the
``KernelOp`` wrapper with a fake kernel and verify that every real factory's
bind/call parameter names exactly cover the kernel's TIR params (no GPU
required -- ``get_tir`` only builds the program).
"""

from __future__ import annotations

import pytest

from rwkv_tl.kernel import (
    cmix_decode,
    cmix_prefill,
    gemv_batch_jit,
    gemv_jit,
    ln_jit,
    ln_per_row_jit,
    tmix_decode,
)
from rwkv_tl.kernel._op import KernelOp, _strip_handle, require_bind
from rwkv_tl.kernel.tmix import _tmix_prefill_back, _tmix_prefill_front


class _FakeParam:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeKernel:
    def __init__(self, names: list[str], out_idx: list[int]) -> None:
        self.prim_func = type("PF", (), {"params": [_FakeParam(n) for n in names]})()
        self.out_idx = out_idx

    def __call__(self, *args):
        return tuple(args)


def _fake_factory(names: list[str], out_idx: list[int]):
    def factory():
        return _FakeKernel(names, out_idx)

    return factory


def test_kernel_op_reorders_args() -> None:
    W, B = object(), object()
    k = KernelOp(
        _fake_factory(["x_handle", "W_handle", "B_handle", "out_handle"], [3]),
        (),
        bind={"W": W, "B": B},
        call=("x",),
        name="ln_kernel",
    )
    assert k("xval") == ("xval", W, B)


def test_kernel_op_call_param_can_be_bound() -> None:
    W = object()
    k = KernelOp(
        _fake_factory(["x_handle", "W_handle", "out_handle"], [2]),
        (),
        bind={"W": W},
        call=("x",),
    )
    assert k("xval") == ("xval", W)


def test_kernel_op_rejects_extra_and_missing() -> None:
    k = KernelOp(
        _fake_factory(["x_handle", "W_handle", "B_handle", "out_handle"], [3]),
        (),
        bind={"W": object(), "B": object()},
        call=("x",),
        name="ln_kernel",
    )
    with pytest.raises(TypeError, match="unexpected arguments"):
        k("x", z=1)
    with pytest.raises(TypeError, match="at most 1 positional"):
        k("x", "y")
    k2 = KernelOp(
        _fake_factory(["x_handle", "W_handle", "out_handle"], [2]),
        (),
        bind={},
        call=("x",),
    )
    with pytest.raises(TypeError, match="missing required arguments"):
        k2("x")


def test_require_bind_validates_names() -> None:
    require_bind({"W": 1, "B": 2}, ["W", "B"], "ln")
    with pytest.raises(TypeError, match="missing weights"):
        require_bind({"W": 1}, ["W", "B"], "ln")
    with pytest.raises(TypeError, match="unexpected weights"):
        require_bind({"W": 1, "B": 2, "Z": 3}, ["W", "B"], "ln")


def test_strip_handle() -> None:
    assert _strip_handle("x_handle") == "x"
    assert _strip_handle("first") == "first"


def _tir_params(factory, *args):
    tir = factory.get_tir(*args)
    out = factory.out_idx if isinstance(factory.out_idx, list) else [factory.out_idx]
    return {_strip_handle(p.name) for i, p in enumerate(tir.params) if i not in out}


FACTORY_CONTRACTS = [
    (ln_jit, (768, "float16"), {"W", "B"}, {"x"}),
    (ln_per_row_jit, (32, 768, "float16"), {"W", "B"}, {"x"}),
    (gemv_jit, (768, 768, "float16"), {"W"}, {"x"}),
    (gemv_batch_jit, (768, 768, 3, "float16"), {"W"}, {"x"}),
    (
        cmix_decode,
        (768, "float16"),
        {"ln_preW", "ln_preB", "x_k", "kWt", "vWt"},
        {"x0", "prev_x"},
    ),
    (
        cmix_prefill,
        (768, "float16", 32),
        {"ln_preW", "ln_preB", "x_k", "kWt", "vWt"},
        {"x0", "prev_x"},
    ),
    (
        tmix_decode,
        (768, "float16", 12, 64, 64, 64, 64),
        {
            "ln_preW",
            "ln_preB",
            "x_rkvwag",
            "rkvWt",
            "v1t",
            "w1t",
            "a1t",
            "g1t",
            "v2t",
            "w2t",
            "a2t",
            "g2t",
            "v0",
            "w0",
            "a0",
            "k_k",
            "k_a",
            "r_k",
            "ln_xW",
            "ln_xB",
            "oWt",
        },
        {"x0", "prev_x", "rnn", "v_first", "first"},
    ),
    (
        _tmix_prefill_front,
        (32, 768, "float16", 12, 64, 64, 64, 64),
        {
            "ln_preW",
            "ln_preB",
            "x_rkvwag",
            "rkvWt",
            "v1t",
            "w1t",
            "a1t",
            "g1t",
            "v2t",
            "w2t",
            "a2t",
            "g2t",
            "v0",
            "w0",
            "a0",
            "k_k",
            "k_a",
        },
        {"x0", "prev_x", "v_first", "first"},
    ),
    (
        _tmix_prefill_back,
        (32, 768, "float16", 12, 64, 64, 64, 64),
        {"r_k", "ln_xW", "ln_xB", "oWt"},
        {"rkv", "w", "kk_norm", "B", "a", "g", "x0", "rnn"},
    ),
]


@pytest.mark.parametrize("factory,args,bind,call", FACTORY_CONTRACTS)
def test_factory_bind_call_match_tir(factory, args, bind, call) -> None:
    assert bind | call == _tir_params(factory, *args), (
        f"{factory.__name__}: bind/call do not cover the TIR params exactly"
    )
