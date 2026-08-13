#!/usr/bin/env python3
"""End-to-end demo of the rwkv_tl user API (single file, stateless).

Walks through the whole workflow:
1. build a model directly from a checkpoint path
2. state tune: prompt -> state, save / load
3. raw logits via ``model.logits``
4. text generation via ``model.generate``
5. low-level token-by-token ``model.decode``

Usage:
    python script/demo_rwkv7.py /path/to/rwkv7-*.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the repo-root package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rwkv_tl


def top_predictions(
    logits: torch.Tensor, model: rwkv_tl.RWKV7Model, k: int = 5
) -> list[tuple[str, float]]:
    """Decode the top-k next tokens with their probabilities."""
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [
        (model.detokenize([int(i)]), float(p))
        for i, p in zip(top.indices.tolist(), top.values.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rwkv_tl end-to-end demo (stateless model API)"
    )
    parser.add_argument("checkpoint", help="Path to an RWKV7 checkpoint (.pth)")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--tune-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--backend",
        choices=["auto", "tl", "torch"],
        default="auto",
        help="'tl' requires CUDA; 'torch' runs anywhere",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--no-graph", action="store_true", help="disable CUDA graphs")
    parser.add_argument(
        "--state",
        type=Path,
        help="load a tuned state from this file instead of tuning",
    )
    parser.add_argument(
        "--save-state", type=Path, help="save the tuned state to this file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend == "tl" and device.type != "cuda":
        parser.error("--backend tl needs CUDA; use 'auto' or 'torch' on CPU")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]

    print(f"[1/5] building model from {args.checkpoint}")
    model = rwkv_tl.rwkv7(
        args.checkpoint,
        device=device,
        dtype=dtype,
        backend=args.backend,
        use_graph=not args.no_graph,
    )

    print("[2/5] state tune (prompt -> state, stateless)")
    if args.state is not None:
        state = rwkv_tl.State.load(args.state, device=device, dtype=dtype)
        print(f"      loaded tuned state from {args.state}")
    else:
        state = model.tune_state(args.tune_prompt)
        if args.save_state is not None:
            state.save(args.save_state)
            print(f"      saved tuned state to {args.save_state}")

    print(f"[3/5] raw logits for {args.prompt!r}")
    logits, state = model.logits(args.prompt, state=state)
    for token, prob in top_predictions(logits, model):
        print(f"      {prob:7.2%}  {token!r}")

    print(f"[4/5] generate {args.max_tokens} tokens")
    out = model.generate(
        args.prompt,
        state=state,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"      {out!r}")

    print("[5/5] low-level decode, one token at a time")
    token = model.encode(" The")[0]
    logits, state = model.decode(torch.tensor([token], device=device), state)
    top = top_predictions(logits, model)[0]
    print(f"      after feeding token {token!r}: next = {top[0]!r} ({top[1]:.2%})")


if __name__ == "__main__":
    main()
