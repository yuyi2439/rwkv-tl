"""Load an RWKV7 checkpoint and print generated text."""

import argparse

from rwkv_tl import rwkv7
from rwkv_tl.core import RWKV7Weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--backend",
        choices=("tl", "torch"),
        required=True,
        help="tl = tilelang (CUDA), torch = pure PyTorch",
    )
    parser.add_argument("--device", default="cuda", help="Device to load the weight on")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    w = RWKV7Weight(args.checkpoint, device=args.device)
    model = rwkv7(w, backend=args.backend)
    print(model.generate(args.prompt, max_new_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
