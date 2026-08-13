#!/usr/bin/env python3
"""Benchmark tilelang vs pure-torch RWKV7 on CUDA (decode + prefill).

Compares three variants of the same checkpoint on GPU, fp16:
  - torch:   rwkv_tl.RWKV7Torch (eager PyTorch, no custom kernels)
  - tl:      rwkv_tl.RWKV7TL (fused tilelang kernels, eager launches)
  - tl+graph:rwkv_tl.RWKV7TL wrapped in CUDAGraph (the default `rwkv7()` path)

Usage:
    python script/bench_tl_vs_torch.py /path/to/rwkv7-0.1b.pth
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

# Make the repo-root package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rwkv_tl import RWKV7TL, RWKV7Torch
from rwkv_tl.core import CUDAGraph, RWKV7Weight

PREFILL_TS = (32, 64, 128, 256, 512)
WARMUP = 2
RUNS = 7
DECODE_STEPS = 64


def median_ms(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def bench(name: str, model, device: torch.device) -> None:
    print(f"\n== {name} ==")

    print(f"  prefill (median of {RUNS}):")
    for T in PREFILL_TS:
        tok = torch.arange(T, dtype=torch.long, device=device)

        def run_prefill(tok=tok):
            model.prefill(tok, model.new_state())

        try:
            ms = median_ms(run_prefill, WARMUP, RUNS)
        except torch.cuda.OutOfMemoryError:
            print(f"    T={T:4d}  OOM")
            break
        print(f"    T={T:4d}  {ms:9.3f} ms  ({T / ms * 1000:8.1f} tok/s)")

    tok1 = torch.tensor([7], dtype=torch.long, device=device)
    ms = median_ms(
        lambda: model.decode(tok1, model.new_state()), WARMUP * 2, DECODE_STEPS
    )
    print(f"  decode: {ms:9.3f} ms/token  ({1000 / ms:8.1f} tok/s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tilelang vs pure-torch RWKV7 benchmark (CUDA)"
    )
    parser.add_argument("checkpoint", help="Path to an RWKV7 checkpoint (.pth)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(
        f"GPU {name} (sm_{major}{minor})  torch {torch.__version__} "
        f"(cuda {torch.version.cuda})"
    )

    w = RWKV7Weight(args.checkpoint, device=device, dtype=torch.float16)
    print(f"checkpoint loaded: L={w.L} C={w.C}")

    torch_model = RWKV7Torch(w, is_torch_compile=False)
    tl_model = RWKV7TL(w, is_torch_compile=False)
    tl_graph = CUDAGraph(RWKV7TL(w, is_torch_compile=False))

    bench("torch (eager)", torch_model, device)
    bench("tl (eager)", tl_model, device)
    bench("tl+graph", tl_graph, device)


if __name__ == "__main__":
    main()
