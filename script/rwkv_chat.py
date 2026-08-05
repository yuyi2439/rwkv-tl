#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

# Make the repo-root packages importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo import make_rwkv7
from rwkv_tl.state import State
from rwkv_tl.tokenizer import Tokenizer
from rwkv_tl.weight import RWKV7Weight


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RWKV chat")
    parser.add_argument(
        "checkpoint",
        help="Path to RWKV checkpoint (.pth)",
    )
    parser.add_argument(
        "vocab",
        help="Path to RWKV vocabulary file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum response tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Softmax temperature (<=0 = greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 = off)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (1.0 = off)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty on generated tokens (1.0 = off)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # tuned (default): device-name-matched CUDA-Graph variant (MX450/RTX3060),
    # falling back to auto (fp16 on sm<80, bf16 otherwise), so the same chat
    # works on any device.
    w = RWKV7Weight(args.checkpoint)
    model_cls = make_rwkv7(w.device)
    model = model_cls(w)
    tokenizer = Tokenizer(args.vocab)
    S = State(
        model.L,
        model.C,
        64,
        device=model.w.device,
    )

    print("Simple RWKV chat. Empty input exits.")
    while True:
        text = input("user: ")
        if text.strip() == "":
            print("Exit.")
            break

        user_tokens = tokenizer.encode(f"User: {text}\n\nAssistant: ")
        response_tokens, S = model.generate(
            user_tokens,
            S,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        response = tokenizer.decode(response_tokens)

        print("assistant:", response)


if __name__ == "__main__":
    with torch.device("cuda"):
        main()
