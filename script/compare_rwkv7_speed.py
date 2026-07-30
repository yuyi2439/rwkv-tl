#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
FAST_ROOT = Path("/home/yuyi2439/rwkv/Albatross/faster3a_2607")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwkv_tl import RWKV7 as ProjectRWKV7


def percentile(values, q):
    if not values:
        raise ValueError("no timing values")
    return float(
        torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0).item()
    )


def load_fast_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("rwkv7_fast_v3a", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_tensors_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in obj)
    return obj


def make_state(model, batch_size: int, device: torch.device):
    zero_state_fn = getattr(model, "zero_state", None)
    if zero_state_fn is None:
        raise AttributeError(f"{type(model).__name__} has no zero_state method")
    try:
        sig = inspect.signature(zero_state_fn)
    except (TypeError, ValueError):
        state = zero_state_fn()
    else:
        if len(sig.parameters) == 0:
            state = zero_state_fn()
        else:
            state = zero_state_fn(batch_size)

    return _move_tensors_to_device(state, device)


def _prepare_tokens(model, tokens: torch.Tensor):
    if model.__class__.__module__ == "rwkv_tl":
        return tokens.reshape(-1)
    return tokens


def bench_case(
    model, batch_size: int, seq_len: int, warmup: int, iters: int, device: torch.device
):
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    state = make_state(model, batch_size, device)
    tokens = torch.arange(batch_size * seq_len, dtype=torch.long, device=device).view(
        batch_size, seq_len
    )
    tokens = (tokens * 1103515245 + 12345) % 65536
    input_tokens = _prepare_tokens(model, tokens)

    for _ in range(warmup):
        _ = model.forward(input_tokens, state)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model.forward(input_tokens, state)
            end.record()
            torch.cuda.synchronize(device=device)
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            _ = model.forward(input_tokens, state)
            times.append((time.perf_counter() - t0) * 1000.0)

    p10 = percentile(times, 10)
    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    tok_s = batch_size * seq_len * 1000.0 / p50
    return p10, p50, p90, tok_s


def build_project_model(checkpoint_path: Path, vocab_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    moved = {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in ckpt.items()
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(moved, tmp.name)
        tmp_path = Path(tmp.name)

    try:
        model = ProjectRWKV7(str(tmp_path), str(vocab_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return model


def build_fast_model(module_path: Path, model_path: str):
    module = load_fast_module(module_path)
    module.MODEL_PATH = model_path  # pyright: ignore[reportAttributeAccessIssue]
    module.load_extensions()
    return module.RWKV7()


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested")

    if args.project_checkpoint is None:
        raise ValueError("--project-checkpoint is required")
    if args.vocab is None:
        raise ValueError("--vocab is required")

    project_model = build_project_model(
        Path(args.project_checkpoint), Path(args.vocab), device
    )
    fast_model = build_fast_model(
        Path(args.fast_script) / "rwkv7_fast_v3a.py", args.project_checkpoint
    )

    print("csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50", flush=True)

    for case in args.cases.split(","):
        case = case.strip()
        if not case:
            continue
        batch_size_str, seq_len_str = case.lower().split("x", 1)
        B = int(batch_size_str)
        T = int(seq_len_str)

        p10, p50, p90, tok_s = bench_case(
            project_model, B, T, args.warmup, args.iters, device
        )
        print(
            f"RESULT label=project_rwkv7 B={B} T={T} iters={args.iters} p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}",
            flush=True,
        )
        print(
            f"csv,project_rwkv7,{B},{T},{args.iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}",
            flush=True,
        )

        p10, p50, p90, tok_s = bench_case(
            fast_model, B, T, args.warmup, args.iters, device
        )
        print(
            f"RESULT label=rwkv7_fast_v3a B={B} T={T} iters={args.iters} p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}",
            flush=True,
        )
        print(
            f"csv,rwkv7_fast_v3a,{B},{T},{args.iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare rwkv7_fast_v3a.py with the project RWKV7 implementation"
    )
    parser.add_argument("--project-checkpoint")
    parser.add_argument("--vocab")
    parser.add_argument("--fast-script")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--cases",
        default="1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if not args.project_checkpoint:
        parser.error("--project-checkpoint is required")
    if not args.vocab:
        parser.error("--vocab is required")

    run_benchmark(args)


if __name__ == "__main__":
    main()
