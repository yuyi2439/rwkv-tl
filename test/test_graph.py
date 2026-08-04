"""Correctness of GraphDecoder against baseline forward.

Verifies that the CUDA Graph-accelerated single-token decoder agrees with the
plain ``model.forward`` prefill path: same argmax / top-5 and a bounded logit
difference. The two paths are not bit-exact because the decode path runs the
fused tilelang kernels per token while prefill runs batched torch ops; the
difference stays at bf16 rounding level (~0.25 on 0.1B).
"""

from __future__ import annotations

import pytest
import torch

from rwkv_tl import RWKV7
from rwkv_tl.graph_decode import GraphDecoder
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]
MAX_ABS_TOL = 4.0


@pytest.fixture(scope="module")
def model(ckpt_path: str) -> RWKV7:
    # Eager instance (see test_forward fixture note).
    with torch.device("cuda"):
        return RWKV7(RWKV7Weight(ckpt_path), is_torch_compile=False)


def _baseline_logits(model: RWKV7, tokens: list[int]) -> torch.Tensor:
    with torch.device("cuda"):
        S = State(
            model.w.N_LAYER,
            model.w.N_EMBD,
            64,
            device=model.emb.device,
        )
        logits = None
        for t in tokens:
            logits, _ = model.decode(t, S)
    assert logits is not None
    return logits.float().cpu()


def _graph_logits(model: RWKV7, tokens: list[int]) -> torch.Tensor:
    dec = GraphDecoder(model)
    dec.reset()
    out = None
    for t in tokens:
        out = dec.step(t)
    assert out is not None
    return out.float().cpu()


def _assert_consistent(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    diff = (got - ref).abs()
    assert diff.max().item() <= MAX_ABS_TOL, (
        f"{label}: max_abs={diff.max().item()} (tol={MAX_ABS_TOL})"
    )
    assert int(got.argmax()) == int(ref.argmax()), (
        f"{label}: argmax mismatch {int(got.argmax())} vs {int(ref.argmax())}"
    )
    top5_got = set(torch.topk(got, 5).indices.tolist())
    top5_ref = set(torch.topk(ref, 5).indices.tolist())
    assert top5_got == top5_ref, f"{label}: top-5 mismatch"


def test_graph_logits_consistent(model: RWKV7) -> None:
    """GraphDecoder logits must agree with the baseline forward within tolerance."""
    ref = _baseline_logits(model, TOKENS)
    got = _graph_logits(model, TOKENS)
    _assert_consistent(got, ref, "graph-vs-baseline")


def test_graph_argmax_match(model: RWKV7) -> None:
    """Argmax of final logits must match between graph and baseline."""
    ref = _baseline_logits(model, TOKENS)
    got = _graph_logits(model, TOKENS)
    assert int(ref.argmax()) == int(got.argmax())
