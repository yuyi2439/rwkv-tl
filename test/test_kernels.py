"""Numerical correctness of the tilelang fused_lerp6 kernel.

Compares the fused kernel against the plain torch LERP form `x + w*(prev-x)`
on bf16/cuda. Since the kernel uses the identical expression with no algorithmic
divergence, the two must be bit-exact.
"""
from __future__ import annotations

import pytest
import torch

from rwkv_tl.kernels import fused_lerp6

N_EMBD = 768
SEED_BASE = 42


def _lerp_ref(x: torch.Tensor, prev: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Plain torch reference: x + w*(prev-x)."""
    return x + w * (prev - x)


@pytest.mark.parametrize("seed", [SEED_BASE + i for i in range(8)])
def test_fused_lerp6_exact(seed: int) -> None:
    """fused_lerp6 output must be bit-identical to 6 separate LERP calls."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N_EMBD, dtype=torch.bfloat16, device="cuda", generator=g)
    prev = torch.randn(N_EMBD, dtype=torch.bfloat16, device="cuda", generator=g)
    weights = [
        torch.randn(N_EMBD, dtype=torch.bfloat16, device="cuda", generator=g)
        for _ in range(6)
    ]

    got = fused_lerp6(x, prev, *weights)
    ref = [_lerp_ref(x, prev, w) for w in weights]

    for name, r, o in zip(("xr", "xw", "xk", "xv", "xa", "xg"), ref, got):
        assert torch.equal(r, o), (
            f"{name} not bit-exact: max_abs={(r-o).abs().max().item()}"
        )
