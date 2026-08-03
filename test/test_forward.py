"""End-to-end forward numerical consistency: rwkv_tl vs pure-torch reference.

Runs the full RWKV7 forward over a fixed 32-token sequence through both the
rwkv_tl implementation (tilelang fused kernels) and the pure-PyTorch reference
(``script/pure_torch_rwkv7.py``), on both the batched-prefill path and the
per-token decode path.

The fused kernels accumulate in fp32 but cast to bf16 at the store and evaluate
gates in a fused kernel rather than discrete torch ops, so the two are not
bit-exact. Acceptance: matching argmax and top-5 token sets, with the logit
difference bounded by a loose tolerance (observed ~0.2 on 0.1B).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from rwkv_tl import RWKV7
from rwkv_tl.model import RWKV7Weight
from rwkv_tl.state import State

# The pure-torch reference lives under script/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "script"))
from pure_torch_rwkv7 import RWKV7Torch

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]
MAX_ABS_TOL = 4.0  # bf16 rounding across 12 recurrent layers stays << this


def _fresh_state(model) -> State:
    return State(
        model.w.N_LAYER,
        model.w.N_EMBD,
        64,
        device=model.emb.device,
    )


def _run_decode(model, tokens) -> torch.Tensor:
    with torch.device("cuda"):
        S = _fresh_state(model)
        logits = None
        for t in tokens:
            logits, S = model.decode(t, S)
    return logits.float().cpu()


def _run_prefill(model, tokens) -> torch.Tensor:
    with torch.device("cuda"):
        S = _fresh_state(model)
        logits, _ = model.prefill(tokens, S)
    return logits.float().cpu()


@pytest.fixture(scope="module")
def models(ckpt_path: str) -> tuple[RWKV7, RWKV7Torch]:
    # Correctness tests run eager (is_torch_compile=False): torch.compile of
    # decode is validated separately on the target GPU (see benchmark --compile).
    with torch.device("cuda"):
        w = RWKV7Weight(ckpt_path)
        return RWKV7(w, is_torch_compile=False), RWKV7Torch(w, is_torch_compile=False)


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


def test_decode_consistent(models) -> None:
    """Per-token decode must match the pure-torch reference (argmax/top-5)."""
    tl, ref = models
    _assert_consistent(_run_decode(tl, TOKENS), _run_decode(ref, TOKENS), "decode")


def test_prefill_consistent(models) -> None:
    """Batched prefill must match the pure-torch reference (argmax/top-5)."""
    tl, ref = models
    _assert_consistent(_run_prefill(tl, TOKENS), _run_prefill(ref, TOKENS), "prefill")


def test_decode_matches_prefill(models) -> None:
    """Decode and prefill paths of the same model must agree."""
    tl, _ = models
    _assert_consistent(
        _run_decode(tl, TOKENS), _run_prefill(tl, TOKENS), "decode-vs-prefill"
    )
