#!/usr/bin/env python3
"""Benchmark multiple RWKV-7 implementations through a shared driver.

Currently implemented targets:
- faster3a_2607: Albatross CUDA implementation.
- rwkv_tl: project fused-kernel implementation.
- pure_torch: pure PyTorch baseline.
- graph_decoder: CUDA Graph single-token decode path for rwkv_tl.

Reserved but not yet implemented targets:
- fla
- FlashRWKV

Defaults run only the three implemented targets above. Additional targets can
be enabled later without changing the benchmark driver shape.

Args via argparse:
    --project-checkpoint: model checkpoint path
    --vocab: vocab file path
    --fast-script: fast implementation directory (contains rwkv7_fast_v3a.py)
    --targets: comma-separated implementation targets
    --device: device for rwkv_tl / pure_torch, cpu | cuda
    --cases: comma-separated BxT cases
    --warmup / --iters: warmup and timing iterations
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = REPO_ROOT / "script"
SRC_ROOT = REPO_ROOT / "src"
for path in (SCRIPT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pure_torch_rwkv7 import RWKV7Torch

from rwkv_tl import RWKV7 as ProjectRWKV7


def percentile(values, q):
    """计算分位数（基于 torch.quantile）。

    Args:
        values (list[float] | tuple[float, ...]): 计时样本（毫秒）。
        q (float): 分位数百分位，0-100。

    Returns:
        float: 对应分位数值。

    Callers:
        - `benchmark_rwkv7.py:bench_case`: 返回 p10/p50/p90。
    """
    if not values:
        raise ValueError("no timing values")
    return float(
        torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0).item()
    )


def load_fast_module(module_path: Path):
    """从指定路径动态加载 fast 实现模块。

    Args:
        module_path (Path): rwkv7_fast_v3a.py 的完整路径。

    Returns:
        module: 加载完毕的模块对象。

    Callers:
        - `benchmark_rwkv7.py:build_fast_model`: 构造 fast 模型时调用。
    """
    spec = importlib.util.spec_from_file_location("rwkv7_fast_v3a", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_tensors_to_device(obj, device: torch.device):
    """递归把嵌套结构中的 tensor 迁移到目标 device。

    Args:
        obj: tensor / dict / list / tuple / 其它。
        device (torch.device): 目标设备。

    Returns:
        与 obj 同结构的对象，tensor 已迁移。

    Callers:
        - `benchmark_rwkv7.py:make_state`: zero_state 后迁移到 device。
    """
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
    """调用模型的 zero_state 并把 state 迁移到目标 device。

    通过反射判断 zero_state 是否接受 batch_size 参数，兼容两种签名。

    Args:
        model: 具备 zero_state 方法的模型。
        batch_size (int): 批大小。
        device (torch.device): 目标设备。

    Returns:
        迁移后的 state。

    Callers:
        - `benchmark_rwkv7.py:bench_case`: 每次计时前重置 state。
    """
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
    """根据模型类型预处理 token 张量。

    rwkv_tl / pure_torch 实现接受一维 token 序列，故展平；
    faster3a_2607 / future batched implementations 保留 [B, T] 形状。

    Args:
        model: 待测模型。
        tokens (torch.Tensor): 原始 [B, T] token 张量。

    Returns:
        torch.Tensor: 处理后的 token。

    Callers:
        - `benchmark_rwkv7.py:bench_case`: 生成输入 token 时调用。
    """
    if model.__class__.__module__ in {"rwkv_tl", "pure_torch_rwkv7"}:
        return tokens.reshape(-1)
    return tokens


def parse_targets(text: str) -> list[str]:
    """Parse comma/space separated benchmark targets."""
    targets = [item.strip() for item in text.replace(",", " ").split() if item.strip()]
    if not targets:
        raise ValueError("no targets specified")
    allowed = {
        "faster3a_2607",
        "rwkv_tl",
        "pure_torch",
        "graph_decoder",
        "fla",
        "FlashRWKV",
    }
    unknown = [target for target in targets if target not in allowed]
    if unknown:
        raise ValueError(f"unknown targets: {unknown}; allowed={sorted(allowed)}")
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    return ordered


def _build_materialized_checkpoint(checkpoint_path: Path, device: torch.device) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    moved = {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in ckpt.items()
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(moved, tmp.name)
        return Path(tmp.name)


def bench_case(
    model, batch_size: int, seq_len: int, warmup: int, iters: int, device: torch.device
):
    """对单个 (B, T) 用例做 warmup + iters 次前向计时。

    Args:
        model: 待测模型。
        batch_size (int): 批大小。
        seq_len (int): 序列长度。
        warmup (int): 预热轮数。
        iters (int): 计时轮数。
        device (torch.device): 运行设备。

    Returns:
        tuple[float, float, float, float]: (p10_ms, p50_ms, p90_ms, tok_s_p50)。

    Callers:
        - `benchmark_rwkv7.py:run_benchmark`: 遍历 cases 时调用。
    """
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


def bench_case_graph_decoder(decoder, seq_len: int, warmup: int, iters: int):
    """Benchmark GraphDecoder over a length-T token sequence.

    GraphDecoder is single-token decode only, so one timed run replays T steps.
    """
    tokens = [int((i * 1103515245 + 12345) % 65536) for i in range(seq_len)]

    torch.cuda.synchronize()
    for _ in range(warmup):
        decoder.reset()
        for token in tokens:
            _ = decoder.step(token)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        decoder.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for token in tokens:
            _ = decoder.step(token)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    p10 = percentile(times, 10)
    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    tok_s = seq_len * 1000.0 / p50
    return p10, p50, p90, tok_s


def build_project_model(checkpoint_path: Path, vocab_path: Path, device: torch.device):
    """加载权重并迁移到目标 device，构造 rwkv_tl.RWKV7 实例。

    RWKV7.__init__ 内部会再次 torch.load，因此采用“迁移后另存临时文件”
    的方式把 tensor 放到目标 device，再传路径给 RWKV7。

    Args:
        checkpoint_path (Path): 原始 .pth 权重路径。
        vocab_path (Path): 词表文件路径。
        device (torch.device): 目标设备。

    Returns:
        ProjectRWKV7: 权重已位于 device 上的模型实例。

    Callers:
        - `benchmark_rwkv7.py:run_benchmark`: 构建 rwkv_tl 模型时调用。
    """
    tmp_path = _build_materialized_checkpoint(checkpoint_path, device)

    try:
        model = ProjectRWKV7(str(tmp_path), str(vocab_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return model


def build_pure_torch_model(checkpoint_path: Path, vocab_path: Path, device: torch.device):
    """Build the pure PyTorch RWKV7 baseline on the requested device."""
    tmp_path = _build_materialized_checkpoint(checkpoint_path, device)

    try:
        model = RWKV7Torch(str(tmp_path), str(vocab_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return model


def build_graph_decoder(checkpoint_path: Path, vocab_path: Path):
    """Build rwkv_tl GraphDecoder on CUDA."""
    from rwkv_tl.graph_decode import GraphDecoder

    model = build_project_model(checkpoint_path, vocab_path, torch.device("cuda"))
    return GraphDecoder(model)


def build_fast_model(module_path: Path, model_path: str):
    """加载 fast 实现模块并构造 RWKV7（CUDA）。

    Args:
        module_path (Path): rwkv7_fast_v3a.py 的完整路径。
        model_path (str): 模型权重路径。

    Returns:
        fast 模块内的 RWKV7 实例。

    Callers:
        - `benchmark_rwkv7.py:run_benchmark`: 构建 fast 模型时调用。
    """
    module = load_fast_module(module_path)
    module.MODEL_PATH = model_path  # pyright: ignore[reportAttributeAccessIssue]
    module.load_extensions()
    return module.RWKV7()


def _print_row(label: str, B: int, T: int, iters: int, p10, p50, p90, tok_s) -> None:
    """打印一行 RESULT 与一行 csv。

    Args:
        label (str): 实现标签。
        B (int): 批大小。
        T (int): 序列长度。
        iters (int): 计时轮数。
        p10/p50/p90 (float): 分位延迟（毫秒）。
        tok_s (float): p50 对应吞吐（token/s）。

    Callers:
        - `benchmark_rwkv7.py:run_benchmark`: 每个 case 输出结果。
    """
    print(
        f"RESULT label={label} B={B} T={T} iters={iters} "
        f"p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}",
        flush=True,
    )
    print(
        f"csv,{label},{B},{T},{iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}",
        flush=True,
    )


def run_benchmark(args):
    """根据 --targets 与 --device 运行选定实现的计时。

    faster3a_2607 / graph_decoder 始终 CUDA；rwkv_tl / pure_torch 运行在 --device。
    fla / FlashRWKV 先保留为占位 target，后续再接入。

    Args:
        args: argparse 解析结果。

    Callers:
        - `benchmark_rwkv7.py:main`: 主入口调用。
    """
    rwkv_device = torch.device(args.device)
    if rwkv_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested")

    if args.project_checkpoint is None:
        raise ValueError("--project-checkpoint is required")
    if args.vocab is None:
        raise ValueError("--vocab is required")

    targets = parse_targets(args.targets)

    print("csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50", flush=True)

    parsed_cases: list[tuple[int, int]] = []
    for case in args.cases.split(","):
        case = case.strip()
        if not case:
            continue
        batch_size_str, seq_len_str = case.lower().split("x", 1)
        parsed_cases.append((int(batch_size_str), int(seq_len_str)))

    for target in targets:
        if target == "faster3a_2607":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "faster3a_2607 requires CUDA, but it is unavailable"
                )
            model = build_fast_model(
                Path(args.fast_script) / "rwkv7_fast_v3a.py", args.project_checkpoint
            )
            device = torch.device("cuda")
            mode = "forward"
        elif target == "rwkv_tl":
            model = build_project_model(
                Path(args.project_checkpoint), Path(args.vocab), rwkv_device
            )
            device = rwkv_device
            mode = "forward"
        elif target == "pure_torch":
            model = build_pure_torch_model(
                Path(args.project_checkpoint), Path(args.vocab), rwkv_device
            )
            device = rwkv_device
            mode = "forward"
        elif target == "graph_decoder":
            if not torch.cuda.is_available():
                raise RuntimeError("graph_decoder requires CUDA, but it is unavailable")
            model = build_graph_decoder(Path(args.project_checkpoint), Path(args.vocab))
            device = torch.device("cuda")
            mode = "decode"
        elif target in {"fla", "FlashRWKV"}:
            raise NotImplementedError(
                f"target '{target}' is reserved but not implemented yet"
            )
        else:
            raise ValueError(f"unknown target: {target}")

        try:
            for B, T in parsed_cases:
                if mode == "decode":
                    if B != 1:
                        print(
                            f"SKIP label={target}({device.type}) B={B} T={T} reason=graph_decoder_requires_B1",
                            flush=True,
                        )
                        continue
                    p10, p50, p90, tok_s = bench_case_graph_decoder(
                        model, T, args.warmup, args.iters
                    )
                else:
                    p10, p50, p90, tok_s = bench_case(
                        model, B, T, args.warmup, args.iters, device
                    )
                _print_row(
                    f"{target}({device.type})", B, T, args.iters, p10, p50, p90, tok_s
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def main():
    """主入口：解析参数并运行 benchmark。

    Callers:
        - 命令行: `python script/benchmark_rwkv7.py`
    """
    default_vocab = str(REPO_ROOT / "asset" / "rwkv_vocab_v20230424.txt")
    parser = argparse.ArgumentParser(description="Benchmark multiple RWKV7 implementations")
    parser.add_argument(
        "--project-checkpoint",
        default=os.environ.get("RWKV_CHECKPOINT_PATH"),
        help="Path to the model checkpoint. Can also be supplied via RWKV_CHECKPOINT_PATH.",
    )
    parser.add_argument("--vocab", default=default_vocab)
    parser.add_argument(
        "--fast-script",
        default=os.environ.get("RWKV_FAST_SCRIPT_PATH"),
        help="Path to the faster3a_2607 directory. Can also be supplied via RWKV_FAST_SCRIPT_PATH.",
    )
    parser.add_argument(
        "--targets",
        default="faster3a_2607,rwkv_tl,pure_torch,graph_decoder",
        help=(
            "Comma/space separated targets: faster3a_2607, rwkv_tl, pure_torch, "
            "graph_decoder, fla, FlashRWKV. Defaults to the implemented targets."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="仅作用于 rwkv_tl 实现: cpu | cuda",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--cases",
        default="1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16",
    )
    args = parser.parse_args()

    if not args.project_checkpoint:
        parser.error("--project-checkpoint is required or RWKV_CHECKPOINT_PATH must be set")
    parsed_targets = parse_targets(args.targets)
    if "faster3a_2607" in parsed_targets and not args.fast_script:
        parser.error("--fast-script is required or RWKV_FAST_SCRIPT_PATH must be set")
    if not args.vocab:
        parser.error("--vocab is required")

    run_benchmark(args)


if __name__ == "__main__":
    main()
