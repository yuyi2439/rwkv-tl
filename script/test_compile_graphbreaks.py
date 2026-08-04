#!/usr/bin/env python3
"""Detect torch.compile graph breaks and verify numerical consistency.

Tests rwkv_tl (custom ops) and pure_torch implementations.
Reports graph break count/locations and compiled-vs-eager consistency.

Usage:
    RWKV_CHECKPOINT_PATH=/path/to/model.pth .venv/bin/python script/test_compile_graphbreaks.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch._dynamo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from demo.rwkv7_fp16 import RWKV7FP16 as RWKV7
from demo.rwkv7_torch import RWKV7Torch
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

CKPT = os.environ.get("RWKV_CHECKPOINT_PATH", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_PREFILL = 32
TOKENS = [(i * 1103515245 + 12345) % 65536 for i in range(N_PREFILL)]


def fresh_state(model) -> State:
    return State(
        model.w.N_LAYER,
        model.w.N_EMBD,
        64,
        device=model.emb.device,
    )


def detect_breaks(fn, args, label):
    torch._dynamo.reset()
    print(f"\n{'=' * 60}")
    print(f"GRAPH BREAKS: {label}")
    print(f"{'=' * 60}")
    try:
        ex = torch._dynamo.explain(fn)(*args)
        print(f"  graphs={ex.graph_count}  breaks={ex.graph_break_count}")
        for gb in getattr(ex, "break_reasons", None) or []:
            print(f"    - {str(gb)[:200]}")
        return ex.graph_break_count
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {type(e).__name__}: {e}")
        return -1


def check_consistency(eager_model, compiled_model, label):
    print(f"\n{'=' * 60}")
    print(f"CONSISTENCY: {label}")
    print(f"{'=' * 60}")

    # Decode: eager vs compiled (identical on non-bf16-native devices where
    # torch.compile is disabled; on sm_80+/AMD this compares the two paths).
    tok_int = TOKENS[0]
    s1 = fresh_state(eager_model)
    s2 = fresh_state(compiled_model)
    with torch.no_grad():
        out_eager, _ = eager_model.decode(tok_int, s1)
    with torch.no_grad():
        out_compiled, _ = compiled_model.decode(tok_int, s2)
    diff = (out_eager.float() - out_compiled.float()).abs()
    print(
        f"  decode:  max_diff={diff.max().item():.6e}  mean_diff={diff.mean().item():.6e}"
    )


def quick_bench(eager_model, compiled_model, label, iters=5):
    print(f"\n{'=' * 60}")
    print(f"BENCH: {label}")
    print(f"{'=' * 60}")

    for path_name, n_tok, fn_name in [
        ("decode", 1, "decode"),
        ("prefill", N_PREFILL, "prefill"),
    ]:
        tok = (
            TOKENS[0]
            if fn_name == "decode"
            else torch.tensor(TOKENS, dtype=torch.long, device=DEVICE)
        )

        # eager
        for _ in range(2):
            s = fresh_state(eager_model)
            eager_model.decode(tok, s) if fn_name == "decode" else eager_model.prefill(
                tok, s
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            s = fresh_state(eager_model)
            eager_model.decode(tok, s) if fn_name == "decode" else eager_model.prefill(
                tok, s
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / iters * 1000

        # compiled (decode only; prefill stays eager)
        if fn_name == "decode":
            compiled = compiled_model.decode
            for _ in range(3):
                s = fresh_state(compiled_model)
                compiled(tok, s)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                s = fresh_state(compiled_model)
                compiled(tok, s)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            comp_ms = (time.perf_counter() - t0) / iters * 1000
            speedup = eager_ms / comp_ms if comp_ms > 0 else float("inf")
            print(
                f"  {path_name}: eager={eager_ms:.2f}ms  compiled={comp_ms:.2f}ms  speedup={speedup:.2f}x"
            )
        else:
            print(f"  {path_name}: eager={eager_ms:.2f}ms  (no compile)")


def main():
    if not CKPT:
        print("ERROR: set RWKV_CHECKPOINT_PATH")
        sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT}")
    print(f"PyTorch: {torch.__version__}")

    with torch.device(DEVICE):
        w = RWKV7Weight(CKPT)
        tl_model = RWKV7(w, is_torch_compile=True)
        tl_eager = RWKV7(w, is_torch_compile=False)
        pure_model = RWKV7Torch(w, is_torch_compile=True)
        pure_eager = RWKV7Torch(w, is_torch_compile=False)

    models = [("rwkv_tl", tl_model), ("pure_torch", pure_model)]
    eagers = {"rwkv_tl": tl_eager, "pure_torch": pure_eager}

    # 1. Graph break detection
    for label, model in models:
        # decode: eager decode closures (custom ops on native-bf16 devices,
        # raw kernels elsewhere); traces the path torch.compile would compile.
        s_d = fresh_state(model)
        detect_breaks(model.decode, (TOKENS[0], s_d), f"{label} decode")

        # prefill: prefill with 32 tokens. Not torch.compile'd (each
        # prompt length would recompile a graph for <1.5x; kept eager), so this
        # traces the eager closures, which still graph-break on the raw kernels.
        tok_p = torch.tensor(TOKENS, dtype=torch.long, device=DEVICE)
        s_p = fresh_state(model)
        detect_breaks(model.prefill, (tok_p, s_p), f"{label} prefill")

    # 2. Numerical consistency
    for label, model in models:
        check_consistency(eagers[label], model, label)

    # 3. Quick benchmark
    for label, model in models:
        quick_bench(eagers[label], model, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
