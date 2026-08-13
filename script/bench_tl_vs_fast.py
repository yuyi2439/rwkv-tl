#!/usr/bin/env python3
"""Benchmark rwkv_tl (tilelang) vs Albatross faster3a_2607 on CUDA.

Measures the same checkpoint on GPU (fp16) for prefill (T sweep) and decode:
  - faster3a_2607: Albatross reference implementation (CUDA extensions)
  - tl:           rwkv_tl.RWKV7TL (fused tilelang kernels, eager launches)
  - tl+graph:     rwkv_tl.RWKV7TL wrapped in CUDAGraph

Before timing, both implementations are checked on the same prompt (argmax /
top-5 with a loose fp16 tolerance). Each target is released before the next
is loaded, to keep MX450 (2GB) free of cross-target memory pressure.

Usage:
    python script/bench_tl_vs_fast.py /path/to/rwkv7-0.1b.pth \
        --fast-path /path/to/faster3a_2607/rwkv7_fast_v3a.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

# Make the repo-root package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rwkv_tl import RWKV7TL, CUDAGraph, RWKV7Weight, Tokenizer

PREFILL_TS = (1, 8, 16, 32, 64, 128, 256, 512)
WARMUP = 2
RUNS = 7
DECODE_STEPS = 64
MAX_ABS_TOL = 16.0  # fp16 fusion differs between implementations


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


def load_fast(module_path: Path, ckpt: str):
    spec = importlib.util.spec_from_file_location("rwkv7_fast_v3a", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_PATH = ckpt
    module.load_extensions()
    return module.RWKV7()


def sanity_check(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Compare tl vs faster3a final logits on a fixed prompt (fp16 tolerance)."""
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
        f"sanity: max_abs={diff.max().item():.3f} "
        f"argmax={'OK' if int(got.argmax()) == int(ref.argmax()) else 'MISMATCH'} "
        f"top5={'OK' if top5_got == top5_ref else 'MISMATCH'} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )


def reference_logits(
    fast_model, tokens: list[int], device: torch.device
) -> torch.Tensor:
    """faster3a final logits for a prompt, kept for the later tl comparison."""
    ids = torch.tensor(tokens, dtype=torch.long, device=device)
    return (
        fast_model.forward(ids.view(1, -1), fast_model.zero_state(1))[0].float().cpu()
    )


def bench(name: str, prefill, decode, device: torch.device) -> None:
    print(f"\n== {name} ==")
    print(f"  prefill (median of {RUNS}):")
    for T in PREFILL_TS:
        try:
            ms = median_ms(lambda T=T: prefill(T), WARMUP, RUNS)
        except torch.cuda.OutOfMemoryError:
            print(f"    T={T:4d}  OOM")
            break
        print(f"    T={T:4d}  {ms:9.3f} ms  ({T / ms * 1000:8.1f} tok/s)")
    ms = median_ms(decode, WARMUP * 2, DECODE_STEPS)
    print(f"  decode: {ms:9.3f} ms/token  ({1000 / ms:8.1f} tok/s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rwkv_tl vs Albatross faster3a_2607 benchmark (CUDA)"
    )
    parser.add_argument("checkpoint", help="Path to an RWKV7 checkpoint (.pth)")
    parser.add_argument(
        "--fast-path",
        type=Path,
        required=True,
        help="Path to Albatross rwkv7_fast_v3a.py (required)",
    )
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

    print(f"\nloading faster3a_2607 from {args.fast_path} ...")
    fast = load_fast(args.fast_path, args.checkpoint)
    fast_tok1 = torch.tensor([[7]], dtype=torch.long, device=device)
    text = "The Eiffel tower is in the city of Paris, and it was built in"
    tokens = Tokenizer().encode(text)
    fast_ref = reference_logits(fast, tokens, device)

    def fast_prefill(T, _model=fast):
        state = _model.zero_state(1)
        tok = torch.arange(T, dtype=torch.long, device=device).view(1, T)
        return _model.forward(tok, state)

    fast_state = fast.zero_state(1)
    bench(
        "faster3a_2607 (sm75)",
        fast_prefill,
        lambda _m=fast, _s=fast_state: _m.forward(fast_tok1, _s),
        device,
    )

    # Free faster3a before loading rwkv_tl weights (2GB MX450).
    del fast, fast_state
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nloading rwkv_tl weights from {args.checkpoint} ...")
    w = RWKV7Weight(args.checkpoint, device=device, dtype=torch.float16)
    print(f"checkpoint loaded: L={w.L} C={w.C}")

    tl = RWKV7TL(w, is_torch_compile=False)
    tl_tok1 = torch.tensor([7], dtype=torch.long, device=device)

    S = tl.new_state()
    got, _ = tl.logits(tokens, S)
    sanity_check(got, fast_ref)

    def tl_prefill(T, _model=tl):
        S = _model.new_state()
        if T == 1:
            return _model.decode(tl_tok1, S)
        tok = torch.arange(T, dtype=torch.long, device=device)
        return _model.prefill(tok, S)

    tl_state = tl.new_state()
    bench(
        "tl (eager)",
        tl_prefill,
        lambda _m=tl, _s=tl_state: _m.decode(tl_tok1, _s),
        device,
    )

    tl_graph = CUDAGraph(RWKV7TL(w, is_torch_compile=False))

    def tl_graph_prefill(T, _model=tl_graph):
        S = _model.new_state()
        if T == 1:
            return _model.decode(tl_tok1, S)
        tok = torch.arange(T, dtype=torch.long, device=device)
        return _model.prefill(tok, S)

    graph_state = tl_graph.new_state()
    bench(
        "tl+graph",
        tl_graph_prefill,
        lambda _m=tl_graph, _s=graph_state: _m.decode(tl_tok1, _s),
        device,
    )


if __name__ == "__main__":
    main()
