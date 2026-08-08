"""Numerical correctness of the tilelang fused_lerp6 kernel.

Compares the fused kernel against the plain torch LERP form `x + w*(prev-x)`
on fp16/cuda. Not bit-exact: tilelang emits a fused fp16 `__hfma` for
`x + w*diff` (one rounding), while torch computes `w*(prev-x)` then `x + ...`
with two roundings. The difference is within 1 ulp of fp16 (max_abs ~0.004
here); assert a small tolerance instead of equality.
"""

from __future__ import annotations

import pytest
import torch

from rwkv_tl.kernel import fused_lerp6

N_EMBD = 768
SEED_BASE = 42
MAX_ABS_TOL = 0.01  # fp16 fma-vs-separate-ops rounding, ~1 ulp of 1.0


def _lerp_ref(x: torch.Tensor, prev: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Plain torch reference: x + w*(prev-x)."""
    return x + w * (prev - x)


@pytest.mark.parametrize("seed", [SEED_BASE + i for i in range(8)])
def test_fused_lerp6_exact(seed: int) -> None:
    """fused_lerp6 must match 6 separate LERP calls within fp16 rounding."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N_EMBD, dtype=torch.float16, device="cuda", generator=g)
    prev = torch.randn(N_EMBD, dtype=torch.float16, device="cuda", generator=g)
    weights = [
        torch.randn(N_EMBD, dtype=torch.float16, device="cuda", generator=g)
        for _ in range(6)
    ]

    got = fused_lerp6(x, prev, *weights)
    ref = [_lerp_ref(x, prev, w) for w in weights]

    for name, r, o in zip(("xr", "xw", "xk", "xv", "xa", "xg"), ref, got):
        diff = (r.float() - o.float()).abs().max().item()
        assert diff <= MAX_ABS_TOL, f"{name} max_abs={diff:.4f} (tol={MAX_ABS_TOL})"
