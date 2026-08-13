"""End-to-end forward numerical consistency: rwkv_tl vs pure-torch reference.

Runs the full RWKV7 forward over a fixed 32-token sequence through both the
rwkv_tl implementation (tilelang fused kernels) and the pure-PyTorch reference
(``rwkv_tl/rwkv7_torch.py``), on both the batched-prefill path and the
per-token decode path.

The fused kernels accumulate in fp32 but cast to bf16 at the store and evaluate
gates in a fused kernel rather than discrete torch ops, so the two are not
bit-exact. Acceptance: matching argmax and top-5 token sets, with the logit
difference bounded by a loose tolerance (observed ~0.2 on 0.1B).
"""

from __future__ import annotations

import pytest
import torch

from rwkv_tl import RWKV7TL as RWKV7
from rwkv_tl import RWKV7Torch
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]
MAX_ABS_TOL = 4.0  # bf16 rounding across 12 recurrent layers stays << this


def _fresh_state(model, dtype: torch.dtype = torch.float16) -> State:
    return State(
        model.w.L,
        model.w.C,
        64,
        device=model.w.device,
        dtype=dtype,
    )


def _run_decode(model, tokens, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    with torch.device("cuda"):
        S = _fresh_state(model, dtype)
        logits = None
        for t in tokens:
            logits, S = model.decode(torch.as_tensor([t], device=model.w.device), S)
    return logits.float().cpu()


def _run_prefill(model, tokens, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    with torch.device("cuda"):
        S = _fresh_state(model, dtype)
        # prefill updates S in place (no logits); single-step the last token
        # to obtain the final logits.
        if len(tokens) > 1:
            S = model.prefill(torch.as_tensor(tokens[:-1], device=model.w.device), S)
        logits, _ = model.decode(
            torch.as_tensor([tokens[-1]], device=model.w.device), S
        )
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


def test_bf16_consistent(ckpt_path: str) -> None:
    """The bf16 model must match the pure-torch reference on bf16 weights.

    Skipped on sm_75 (MX450): no native bf16 tensor cores; validate on sm_80+.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if (major, _) < (8, 0):
            pytest.skip("bf16 needs sm_80+ (no bf16 tensor cores on this GPU)")
    with torch.device("cuda"):
        w = RWKV7Weight(ckpt_path, dtype=torch.bfloat16)
        tl = RWKV7(w, is_torch_compile=False)
        ref = RWKV7Torch(w, is_torch_compile=False)
    _assert_consistent(
        _run_decode(tl, TOKENS, torch.bfloat16),
        _run_decode(ref, TOKENS, torch.bfloat16),
        "bf16-decode",
    )
    _assert_consistent(
        _run_prefill(tl, TOKENS, torch.bfloat16),
        _run_prefill(ref, TOKENS, torch.bfloat16),
        "bf16-prefill",
    )
