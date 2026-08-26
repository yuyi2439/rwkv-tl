#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

# Make the repo-root packages importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rwkv_tl
from rwkv_tl.core import RWKV7Weight


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RWKV chat")
    parser.add_argument(
        "checkpoint",
        help="Path to RWKV checkpoint (.pth)",
    )
    parser.add_argument(
        "--backend",
        choices=("tl", "torch"),
        required=True,
        help="tl = tilelang (CUDA), torch = pure PyTorch",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to load the weight on",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt for the conversation",
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
    w = RWKV7Weight(args.checkpoint, device=args.device)
    model = rwkv_tl.rwkv7(w, backend=args.backend)
    messages = [{"role": "system", "content": args.system}]

    print("Simple RWKV chat. Empty input exits.")
    while True:
        text = input("user: ")
        if text.strip() == "":
            print("Exit.")
            break

        messages.append({"role": "user", "content": text})
        response = model.chat(
            messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        messages.append({"role": "assistant", "content": response})

        print("assistant:", response)


if __name__ == "__main__":
    with torch.device("cuda"):
        main()
