"""Numerical correctness of the neo TMIX decode kernel.

Compares ``tmix_decode`` (single-token fused time-mix) against the pure-torch
reference ``rwkv7_torch.time_mix``. Requires ``RWKV_CHECKPOINT_PATH``.
fp16 on cuda; tolerances are ~fp16 ULP over the whole chain.
"""

from __future__ import annotations

import os

import pytest
import torch

from demo.rwkv7_torch import time_mix as time_mix_ref
from rwkv_tl.kernel.tmix import tmix_decode
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

CKPT = os.environ.get("RWKV_CHECKPOINT_PATH")
if not CKPT:
    pytest.skip("RWKV_CHECKPOINT_PATH not set", allow_module_level=True)

MAX_ABS_TOL = 0.05  # whole-chain fp16 accumulation
RNN_TOL = 0.05


@pytest.fixture(scope="module")
def model():
    return RWKV7Weight(CKPT, dtype=torch.float16)


@pytest.fixture(scope="module")
def kernel(model):
    b = model.blocks[0].att
    C, H, N = model.C, b.r_k.shape[0], b.r_k.shape[1]
    Rv, Rw, Ra, Rg = b.v1.shape[1], b.w1.shape[1], b.a1.shape[1], b.g1.shape[1]
    return tmix_decode(C, "float16", H, Rv, Rw, Ra, Rg)


@pytest.fixture(scope="module")
def kw(model):
    b = model.blocks[0].att
    C = model.C
    d = {
        "ln_preW": b.ln_pre.w,
        "ln_preB": b.ln_pre.b,
        "x_r": b.x_r,
        "x_w": b.x_w,
        "x_k": b.x_k,
        "x_v": b.x_v,
        "x_a": b.x_a,
        "x_g": b.x_g,
        "rWt": b.rkvWt[0],
        "kWt": b.rkvWt[1],
        "vWt": b.rkvWt[2],
        "v1t": b.v1t,
        "w1t": b.w1t,
        "a1t": b.a1t,
        "g1t": b.g1t,
        "v2t": b.v2.T.contiguous(),
        "w2t": b.w2.T.contiguous(),
        "a2t": b.a2.T.contiguous(),
        "g2t": b.g2.T.contiguous(),
        "v0": b.v0.reshape(-1),
        "w0": b.w0.reshape(-1),
        "a0": b.a0.reshape(-1),
        "k_k": b.k_k.reshape(-1),
        "k_a": b.k_a.reshape(-1),
        "r_k": b.r_k,
        "ln_xW": b.ln_x.w,
        "ln_xB": b.ln_x.b,
        "oWt": b.oWt,
    }
    return {kk: vv.contiguous() for kk, vv in d.items()}


@pytest.mark.parametrize("seed", [42, 43])
def test_tmix_decode(seed: int, model, kernel, kw) -> None:
    C = model.C
    b = model.blocks[0].att
    H, N = b.r_k.shape[0], b.r_k.shape[1]

    S_ref = State(1, C, N, device="cuda", dtype=torch.float16)
    S_k = State(1, C, N, device="cuda", dtype=torch.float16)
    st = S_k.tmix[0]
    v_first_k = torch.zeros(C, device="cuda", dtype=torch.float16)
    v_first_e = None

    for t in range(4):
        g = torch.Generator(device="cuda").manual_seed(seed * 10 + t)
        x0 = torch.randn(C, device="cuda", dtype=torch.float16, generator=g) * 0.5

        ref, v_first_e = time_mix_ref(
            b, x0, v_first_e, S_ref.tmix[0], H, N
        )
        out = kernel(
            x0,
            kw["ln_preW"],
            kw["ln_preB"],
            kw["x_r"],
            kw["x_w"],
            kw["x_k"],
            kw["x_v"],
            kw["x_a"],
            kw["x_g"],
            kw["rWt"],
            kw["kWt"],
            kw["vWt"],
            kw["v1t"],
            kw["w1t"],
            kw["a1t"],
            kw["g1t"],
            kw["v2t"],
            kw["w2t"],
            kw["a2t"],
            kw["g2t"],
            kw["v0"],
            kw["w0"],
            kw["a0"],
            kw["k_k"],
            kw["k_a"],
            kw["r_k"],
            kw["ln_xW"],
            kw["ln_xB"],
            kw["oWt"],
            st["x"],
            st["rnn"],
            v_first_k,
            1 if t == 0 else 0,
        )

        err = (out.float() - ref.float()).abs().max().item()
        assert err <= MAX_ABS_TOL, f"step {t}: out max_abs={err:.4f}"
        rnn_err = (st["rnn"] - S_ref.tmix[0]["rnn"]).abs().max().item()
        assert rnn_err <= RNN_TOL, f"step {t}: rnn max_abs={rnn_err:.4f}"
        assert (st["x"] - S_ref.tmix[0]["x"]).abs().max().item() <= 0.01
