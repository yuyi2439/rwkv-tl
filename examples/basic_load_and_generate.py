"""Load an RWKV7 checkpoint and print generated text."""

import argparse

from rwkv_tl import rwkv7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    model = rwkv7(args.checkpoint)
    print(model.generate(args.prompt, max_new_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
