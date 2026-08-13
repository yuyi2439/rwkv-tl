"""Load a checkpoint and chat with it (chat template + generate)."""

import argparse

from rwkv_tl import rwkv7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model = rwkv7(args.checkpoint)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(model.chat(messages, max_new_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
