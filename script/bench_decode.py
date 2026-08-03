#!/usr/bin/env python3
"""Compare eager decode/prefill latency of rwkv_tl vs the pure-torch reference.

Focuses on the eager (uncompiled) path, which is what runs on devices without
native bf16 (sm_75) or before torch.compile kicks in. State is zeroed in-place
via ``reset_state`` so the measurement excludes allocation overhead.

Usage:
    .venv/bin/python script/bench_decode.py --checkpoint <path> [--vocab <path>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "script"))

from pure_torch_rwkv7 import RWKV7Torch

from rwkv_tl import RWKV7

N_TOKENS = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_TOKENS)]


def bench_decode(model, iters: int = 10) -> float:
    """ms per 32-token decode sequence (eager, in-place state reset)."""
    with torch.device(model.emb.device):
        S = model.zero_state()
    for _ in range(3):
        model.reset_state(S)
        for t in TOKENS:
            model._eager_run_one(t, S)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model.reset_state(S)
        for t in TOKENS:
            model._eager_run_one(t, S)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_prefill(model, iters: int = 10) -> float:
    """ms per 32-token prefill (eager, in-place state reset)."""
    tok = torch.tensor(TOKENS, dtype=torch.long, device=model.emb.device)
    with torch.device(model.emb.device):
        S = model.zero_state()
    for _ in range(3):
        model.reset_state(S)
        model.forward_prefill(tok, S)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model.reset_state(S)
        model.forward_prefill(tok, S)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _decode_logits(model) -> torch.Tensor:
    """Last-token logits from per-token eager decode over TOKENS."""
    with torch.device(model.emb.device):
        S = model.zero_state()
    logits = None
    for t in TOKENS:
        logits, S = model._eager_run_one(t, S)
    assert logits is not None
    return logits.reshape(-1).float().cpu()


def _prefill_logits(model) -> torch.Tensor:
    """Last-token logits from batched prefill over TOKENS."""
    tok = torch.tensor(TOKENS, dtype=torch.long, device=model.emb.device)
    with torch.device(model.emb.device):
        S = model.zero_state()
    logits, _ = model.forward_prefill(tok, S)
    return logits.reshape(-1).float().cpu()


def _check_correctness(
    got: torch.Tensor, ref: torch.Tensor, label: str, tol: float
) -> None:
    """Verify ``got`` matches the reference; raise SystemExit on mismatch."""
    if got.shape != ref.shape:
        raise SystemExit(
            f"SKIP {label}: shape mismatch {tuple(got.shape)} vs {tuple(ref.shape)}"
        )
    diff = (got - ref).abs().max().item()
    argmax_ok = int(got.argmax()) == int(ref.argmax())
    if not (argmax_ok and diff <= tol):
        raise SystemExit(
            f"SKIP {label}: incorrect (argmax {'ok' if argmax_ok else 'MISMATCH'} "
            f"max_abs={diff:.4f}, tol={tol}); not reporting latency"
        )


def main():
    default_vocab = str(REPO / "asset" / "rwkv_vocab_v20230424.txt")
    parser = argparse.ArgumentParser(
        description="Eager decode/prefill latency comparison"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("RWKV_CHECKPOINT_PATH"),
        help="Model checkpoint (.pth); defaults to RWKV_CHECKPOINT_PATH.",
    )
    parser.add_argument("--vocab", default=default_vocab)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--no-correctness-check",
        action="store_true",
        help="Disable the rwkv_tl-vs-pure_torch correctness gate.",
    )
    parser.add_argument(
        "--correctness-tol",
        type=float,
        default=16.0,
        help="Max-abs logit tolerance for the correctness gate (default 16.0).",
    )
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required or set RWKV_CHECKPOINT_PATH")

    with torch.device("cuda"):
        tl = RWKV7(args.checkpoint, args.vocab)
        pt = RWKV7Torch(args.checkpoint, args.vocab)

    print(f"device: {tl.emb.device}")
    print(f"rwkv_tl use_custom_ops (decode dispatch): {tl._use_custom_ops}")

    # Correctness gate: refuse to report latency for broken output.
    if not args.no_correctness_check:
        _check_correctness(
            _decode_logits(tl), _decode_logits(pt), "decode", args.correctness_tol
        )
        _check_correctness(
            _prefill_logits(tl), _prefill_logits(pt), "prefill", args.correctness_tol
        )
        print("correctness: decode/prefill match pure_torch reference")

    for label, model in (("rwkv_tl", tl), ("pure_torch", pt)):
        print(
            f"{label}: decode {bench_decode(model, args.iters):8.2f} ms/32tok  "
            f"prefill {bench_prefill(model, args.iters):8.2f} ms/32tok"
        )


if __name__ == "__main__":
    main()
