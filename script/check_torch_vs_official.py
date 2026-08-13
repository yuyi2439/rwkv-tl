#!/usr/bin/env python3
"""Validate rwkv_tl's pure-torch model against the official RWKV-LM v7 demo.

Loads the official ``rwkv_v7_demo.py`` (RWKV-LM/RWKV-v7) with its pure-torch
path (``USE_CUDA_KERNEL=False``) and compares logits with
``rwkv_tl.RWKV7Torch`` on the same checkpoint, for both the batched path and
the per-token decode path:

    python script/check_torch_vs_official.py /path/to/rwkv7-0.1b.pth

The official demo is a torch.jit.script (GPT-mode) model, so this script only
imports the class definitions from it -- its module-level inference tail is
not executed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import torch

# Make the repo-root package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rwkv_tl

DEFAULT_FAST_PATH = "/home/yuyi2439/rwkv/RWKV-LM/RWKV-v7/rwkv_v7_demo.py"
MAX_ABS_TOL = 2.0  # fp16 op ordering differs between the two implementations

TEXTS = [
    "The Eiffel tower is in the city of",
    "Hello, world! The quick brown fox jumps over",
    "def fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n - 1)",
]


def load_official(demo_path: Path, ckpt_path: str) -> tuple[object, object]:
    """Build the official RWKV7 demo model (pure torch) from its source."""
    src = demo_path.read_text(encoding="utf-8")
    # Stop before the module-level inference tail (model load + LAMBADA run).
    marker = 'model_params = torch.load(MODEL_PATH, map_location="cpu")'
    head, _, _ = src.partition(marker)
    if not head:
        raise ValueError(f"unexpected rwkv_v7_demo.py layout in {demo_path}")

    head = head.replace("USE_CUDA_KERNEL = True", "USE_CUDA_KERNEL = False")
    vocab_path = str(files("rwkv_tl").joinpath("rwkv_vocab_v20230424.txt"))
    head = head.replace(
        'RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")',
        f"RWKV_TOKENIZER({vocab_path!r})",
    )

    z = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    head_size = z["blocks.0.att.r_k"].shape[1]
    # torch.jit resolves module globals at compile time; inline the head size
    # literal so the dynamically imported official demo compiles cleanly.
    head = head.replace("H = C // HEAD_SIZE", f"H = C // {head_size}")
    head = head.replace("N = HEAD_SIZE", f"N = {head_size}")
    # Same for the DTYPE global used inside the jit-compiled RWKV7_OP.
    head = head.replace("DTYPE = torch.half", "__DT_ASGN__")
    head = head.replace("DTYPE", "torch.half")
    head = head.replace("__DT_ASGN__", "DTYPE = torch.half")

    # torch.jit.script needs real source files, so write the (truncated)
    # official demo to a temp module and import it normally.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(head)
        temp_path = f.name
    spec = importlib.util.spec_from_file_location("rwkv_v7_demo_official", temp_path)
    assert spec is not None and spec.loader is not None
    official_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_mod)
    ns = vars(official_mod)

    # Override the 0.1B-hardcoded constants with checkpoint dims so any
    # RWKV7 checkpoint works.
    args = ns["args"]
    args.n_layer = 1 + max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
    args.n_embd = z["emb.weight"].shape[-1]
    args.vocab_size = z["emb.weight"].shape[0]
    _, N = z["blocks.0.att.r_k"].shape
    args.head_size_a = N
    ns["D_DECAY_LORA"] = z["blocks.0.att.w1"].shape[1]
    ns["D_AAA_LORA"] = z["blocks.0.att.a1"].shape[1]
    ns["D_MV_LORA"] = z["blocks.0.att.v1"].shape[1]
    ns["D_GATE_LORA"] = z["blocks.0.att.g1"].shape[1]

    model = ns["RWKV"](args).to(dtype=torch.half)
    model.load_state_dict(z, strict=False)
    return model.cuda(), z


def compare_logits(got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    """Assert logits agreement and print the numbers."""
    got = got.float().cpu()
    ref = ref.float().cpu()
    diff = (got - ref).abs()
    top5_got = set(torch.topk(got, 5).indices.tolist())
    top5_ref = set(torch.topk(ref, 5).indices.tolist())
    ok = (
        diff.max().item() <= MAX_ABS_TOL
        and int(got.argmax()) == int(ref.argmax())
        and top5_got == top5_ref
    )
    print(
        f"  {label:34s} max_abs={diff.max().item():.4f} "
        f"argmax={'OK' if int(got.argmax()) == int(ref.argmax()) else 'MISMATCH'} "
        f"top5={'OK' if top5_got == top5_ref else 'MISMATCH'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        raise AssertionError(f"{label}: logits disagree with the official demo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rwkv_tl.RWKV7Torch vs official RWKV-LM v7 demo"
    )
    parser.add_argument("checkpoint", help="Path to an RWKV7 checkpoint (.pth)")
    parser.add_argument(
        "--fast-path",
        type=Path,
        default=Path(DEFAULT_FAST_PATH),
        help="Path to the official rwkv_v7_demo.py",
    )
    parser.add_argument(
        "--device", default="cuda", help="torch device for the comparison"
    )
    args = parser.parse_args()

    print(f"loading official demo from {args.fast_path} ...")
    official, _ = load_official(args.fast_path, args.checkpoint)
    official.eval()

    print(f"building rwkv_tl.RWKV7Torch from {args.checkpoint} ...")
    model = rwkv_tl.RWKV7Torch(args.checkpoint, device=args.device, dtype=torch.float16)

    with torch.no_grad():
        # Batched path: official GPT-mode over the whole sequence vs our
        # prefill(tokens[:-1]) + decode(last).
        for text in TEXTS:
            tokens = model.encode(text)
            label = f"batched[{len(tokens):3d} tok]"
            ref = official(
                torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)
            )[0, -1]
            got, _ = model.logits(tokens, model.new_state())
            compare_logits(got, ref, label)

        # Decode path: official one token per fresh call vs our carried state.
        tokens = model.encode(TEXTS[0])
        S = model.new_state()
        seq = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)
        for i, t in enumerate(tokens):
            got, S = model.decode(
                torch.tensor([t], dtype=torch.long, device=args.device), S
            )
            # Official GPT-mode is stateless per call: run it over the whole
            # prefix and take the last position's logits.
            ref = official(seq[:, : i + 1])[0, -1]
            if i == len(tokens) - 1 or i % 4 == 0:
                compare_logits(got, ref, f"decode step {i:3d}")

    print("all comparisons PASS: rwkv_tl.RWKV7Torch matches the official demo")


if __name__ == "__main__":
    main()
