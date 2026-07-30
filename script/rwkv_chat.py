#!/usr/bin/env python3

import argparse

import torch

from rwkv_tl import RWKV7


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RWKV greedy chat")
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
    return parser.parse_args()


def greedy_generate(model, S, logits, max_tokens: int):
    tokens = []
    for _ in range(max_tokens):
        token = int(torch.argmax(logits))
        tokens.append(token)
        logits, S = model.forward([token], S)
    return tokens, S


def main():
    args = parse_args()
    model = RWKV7(args.checkpoint, args.vocab)
    S = model.zero_state()

    print("Simple RWKV chat. Empty input exits.")
    while True:
        text = input("user: ")
        if text.strip() == "":
            print("Exit.")
            break

        user_tokens = model.encode(text + "\n")
        logits, S = model.forward(user_tokens, S)
        response_tokens, S = greedy_generate(model, S, logits, args.max_tokens)
        response = model.decode(response_tokens)

        print("assistant:", response)


if __name__ == "__main__":
    with torch.device("cuda"):
        main()
