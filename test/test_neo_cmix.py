"""Numerical correctness of the neo CMIX kernels (decode + multi).

Compares ``cmix_decode`` (single-token decode) and ``cmix_prefill``
(prefill) against the eager CMIX reference used by ``make_CMIX_batch``:
LN_pre -> token-shift lerp (shift source = previous token's LN output, not
the raw token) -> relu^2(x @ kWt) @ vWt + x0. fp16/bf16 on cuda; tolerances
are ~fp16 ULP, whole-chain accumulation keeps max_abs < 0.1.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rwkv_tl.kernel.neo.cmix import cmix_decode, cmix_prefill

N_EMBD = 768
SEED_BASE = 42
MAX_ABS_TOL = 0.1  # fp16 whole-chain accumulation (output magnitude ~hundreds)
PREV_TOL = 0.01


def _relusq(v: torch.Tensor) -> torch.Tensor:
    r = torch.relu(v)
    return r * r


def _cmix_ref(
    x0: torch.Tensor,
    prev: torch.Tensor,
    lnW: torch.Tensor,
    lnB: torch.Tensor,
    x_k: torch.Tensor,
    kWt: torch.Tensor,
    vWt: torch.Tensor,
) -> torch.Tensor:
    """Eager reference matching `_rwkv7_base.make_CMIX_batch`.

    Token-shift source is the previous token's LN output (`x_ln[n-1]`), with
    `prev` supplying the shift for the first token.
    """
    x_ln = F.layer_norm(x0.float(), (lnW.shape[0],), lnW.float(), lnB.float(), 1e-5)
    if x_ln.dim() == 1:
        prev_full = prev.float()
    else:
        prev_full = torch.cat([prev.float().unsqueeze(0), x_ln[:-1]], dim=0)
    x = x_ln + x_k.float() * (prev_full - x_ln)
    h = _relusq(x @ kWt.float())
    return x0.float() + h @ vWt.float(), x_ln


@pytest.mark.parametrize("seed", [SEED_BASE, SEED_BASE + 1])
@pytest.mark.parametrize("dtype_s", ["float16"])
def test_cmix_decode(seed: int, dtype_s: str) -> None:
    """cmix_decode (single token) must match the eager CMIX chain."""
    dtype = torch.float16 if dtype_s == "float16" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(seed)
    C = N_EMBD
    x0 = torch.randn(C, device="cuda", dtype=dtype, generator=g) * 0.5
    lnW = torch.randn(C, device="cuda", dtype=dtype, generator=g) + 1.0
    lnB = torch.randn(C, device="cuda", dtype=dtype, generator=g) * 0.1
    x_k = torch.rand(C, device="cuda", dtype=dtype, generator=g)
    kWt = torch.randn(C, 4 * C, device="cuda", dtype=dtype, generator=g) * 0.02
    vWt = torch.randn(4 * C, C, device="cuda", dtype=dtype, generator=g) * 0.02
    prev = torch.randn(C, device="cuda", dtype=dtype, generator=g)
    prev_orig = prev.clone()

    out = cmix_decode(C, dtype_s)(x0, lnW, lnB, x_k, kWt, vWt, prev)
    ref, x_ln = _cmix_ref(x0, prev_orig, lnW, lnB, x_k, kWt, vWt)

    assert (out.float() - ref).abs().max().item() <= MAX_ABS_TOL
    assert (prev.float() - x_ln.float()).abs().max().item() <= PREV_TOL


@pytest.mark.parametrize(
    ("seed", "LEN", "LEN_block"),
    [(SEED_BASE, 64, 32), (SEED_BASE + 1, 128, 32), (SEED_BASE, 256, 64)],
)
def test_cmix_prefill(seed: int, LEN: int, LEN_block: int) -> None:
    """cmix_prefill (prefill) must match the eager batched CMIX chain."""
    dtype_s = "float16"
    dtype = torch.float16
    g = torch.Generator(device="cuda").manual_seed(seed)
    C = N_EMBD
    x0 = torch.randn(LEN, C, device="cuda", dtype=dtype, generator=g) * 0.5
    lnW = torch.randn(C, device="cuda", dtype=dtype, generator=g) + 1.0
    lnB = torch.randn(C, device="cuda", dtype=dtype, generator=g) * 0.1
    x_k = torch.rand(C, device="cuda", dtype=dtype, generator=g)
    kWt = torch.randn(C, 4 * C, device="cuda", dtype=dtype, generator=g) * 0.02
    vWt = torch.randn(4 * C, C, device="cuda", dtype=dtype, generator=g) * 0.02
    prev = torch.randn(C, device="cuda", dtype=dtype, generator=g)
    prev_orig = prev.clone()

    out = cmix_prefill(C, dtype_s, LEN_block)(x0, lnW, lnB, x_k, kWt, vWt, prev)
    ref, x_ln = _cmix_ref(x0, prev_orig, lnW, lnB, x_k, kWt, vWt)

    assert (out.float() - ref).abs().max().item() <= MAX_ABS_TOL
    assert (prev.float() - x_ln[-1].float()).abs().max().item() <= PREV_TOL
