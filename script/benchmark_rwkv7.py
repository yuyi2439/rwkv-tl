#!/usr/bin/env python3
"""Benchmark multiple RWKV7 implementations through a shared driver.

Models are built with ``demo.make_rwkv7(backend=...)`` (no bespoke builder
functions) so every target goes through the same entry point:

- faster3a_2607: Albatross CUDA implementation (external module).
- tl-fp16: tilelang fp16 (``demo.make_rwkv7(backend="fp16")``).
- tl-bf16: tilelang bf16 (raw checkpoint dtype, ``backend="bf16"``).
- tl-tuned: per-device tuned variant (``backend="tuned"``).
- pure-torch: pure PyTorch baseline (``backend="torch"``).

Reserved but not yet implemented targets:
- fla
- FlashRWKV

Args via argparse:
    --project-checkpoint: model checkpoint path
    --vocab: vocab file path
    --fast-script: fast implementation directory (contains rwkv7_fast_v3a.py)
    --targets: comma-separated implementation targets
    --device: device for the project targets, cpu | cuda
    --cases: comma-separated BxT cases
    --warmup / --iters: warmup and timing iterations
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = REPO_ROOT / "script"
for path in (SCRIPT_ROOT, SRC_ROOT := REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo import RWKV7Model, make_rwkv7
from rwkv_tl.state import State
from rwkv_tl.weight import RWKV7Weight

BACKEND_FOR_TARGET = {
    "tl-fp16": "fp16",
    "tl-bf16": "bf16",
    "tl-mx450": "mx450",
    "tl-rtx3060": "rtx3060",
    "tl-tuned": "tuned",
    "pure-torch": "torch",
}

DTYPE_FOR_TARGET = {
    "tl-fp16": torch.float16,
    "tl-bf16": torch.bfloat16,
    "tl-mx450": torch.float16,
    "tl-rtx3060": torch.float16,
    "tl-tuned": torch.float16,
    "pure-torch": torch.float16,
}

# Project targets gated against a matching-dtype pure-torch reference.
GATED_TARGETS = {"tl-fp16", "tl-bf16", "tl-mx450", "tl-rtx3060", "tl-tuned"}


def percentile(values, q):
    """计算分位数（基于 torch.quantile）。

    Args:
        values (list[float] | tuple[float, ...]): 计时样本（毫秒）。
        q (float): 分位数百分位，0-100。

    Returns:
        float: 对应分位数值。
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
    """
    spec = importlib.util.spec_from_file_location("rwkv7_fast_v3a", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrectnessError(RuntimeError):
    """Raised when a target's output does not match the pure-torch reference.

    The benchmark catches this to SKIP the case instead of reporting a latency
    for broken code.
    """


def make_state(model, batch_size: int | None = None) -> State | list[torch.Tensor]:
    """构造该模型的零状态。

    State 与 Model 解耦（Model 无状态），故本项目模型直接按属性构造
    （device + dtype 感知）。第三方 target（faster3a_2607）不实现 RWKV7Model，
    其状态由其自身 ``zero_state(B)`` 提供。

    Args:
        model: 待测模型。
        batch_size (int | None): 仅第三方模型创建状态时使用（``zero_state(B)``）。

    Returns:
        State（本项目）或第三方模型的零状态。
    """
    if isinstance(model, RWKV7Model):
        return State(
            model.w.L,
            model.w.C,
            64,
            device=model.w.device,
            dtype=model.w.dtype,
        )
    return model.zero_state(batch_size or 1)


def _fresh_logits(
    model, input_tokens: torch.Tensor, batch_size: int | None = None
) -> torch.Tensor:
    """Run the model on a fresh state; return flattened final logits.

    Handles both ``forward -> (logits, state)`` (project implementations) and
    ``forward -> logits`` (faster3a_2607).
    """
    state = make_state(model, batch_size)
    out = model.forward(input_tokens, state)
    logits = out[0] if isinstance(out, tuple) else out
    return logits.reshape(-1).float()


# Top-2 logit gap below which argmax is considered a near-tie (unreliable
# between correct implementations); not reported as a correctness failure.
NEAR_TIE_GAP = 0.05


def _check_correctness(
    got: torch.Tensor, ref: torch.Tensor, label: str, tol: float
) -> None:
    """Verify ``got`` matches the reference; raise CorrectnessError otherwise.

    A large logit diff (> tol) always fails. When the diff is within tol but
    argmax differs, fail only if the top-2 logits are clearly separated (gap >
    NEAR_TIE_GAP): a near-tie can flip argmax between two correct fp16/fp32
    implementations (accumulation order), so it is not evidence of breakage.
    """
    if got.shape != ref.shape:
        raise CorrectnessError(
            f"{label}: shape mismatch {tuple(got.shape)} vs {tuple(ref.shape)}"
        )
    diff = (got - ref).abs().max().item()
    if diff > tol:
        raise CorrectnessError(f"{label}: max_abs={diff:.4f} (tol={tol})")
    if int(got.argmax()) == int(ref.argmax()):
        return
    top2 = torch.topk(got, 2).values
    gap = float(top2[0] - top2[1])
    if gap > NEAR_TIE_GAP:
        raise CorrectnessError(
            f"{label}: argmax MISMATCH max_abs={diff:.4f} (tol={tol})"
        )


def _prepare_tokens(model, tokens: torch.Tensor) -> torch.Tensor:
    """根据模型类型预处理 token 张量。

    RWKV7Model 实现接受一维 token 序列，故展平；faster3a_2607 保留 [B, T]。
    """
    if isinstance(model, RWKV7Model):
        return tokens.reshape(-1)
    return tokens


def parse_targets(text: str) -> list[str]:
    """Parse comma/space separated benchmark targets."""
    targets = [item.strip() for item in text.replace(",", " ").split() if item.strip()]
    if not targets:
        raise ValueError("no targets specified")
    allowed = {
        "faster3a_2607",
        "tl-fp16",
        "tl-bf16",
        "tl-mx450",
        "tl-rtx3060",
        "tl-tuned",
        "pure-torch",
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


def _eager_dispatch(
    model, input_tokens: torch.Tensor, state: State | list[torch.Tensor]
):
    """Dispatch to the eager decode / prefill when available.

    Project instances are built with is_torch_compile=False, so their
    decode/prefill are already eager and routing through them (rather
    than `model.forward`) avoids recompiling a fresh graph for every distinct
    token count (T) inside a benchmark sweep -- minutes per case with the GPU
    idle. faster3a keeps `model.forward`.
    """
    if isinstance(model, RWKV7Model):
        assert isinstance(state, State)
        if input_tokens.numel() == 1:
            return model.decode(input_tokens, state)
        return model.prefill(input_tokens, state)
    return model.forward(input_tokens, state)


def bench_case(
    model,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    device: torch.device,
    reference=None,
    correctness_tol: float | None = None,
):
    """对单个 (B, T) 用例做 warmup + iters 次前向计时。

    Args:
        model: 待测模型。
        batch_size (int): 批大小。
        seq_len (int): 序列长度。
        warmup (int): 预热轮数。
        iters (int): 计时轮数。
        device (torch.device): 运行设备。
        reference: 正确性门控的参考模型（pure_torch）；None 表示跳过门控。
        correctness_tol: 门控的 max_abs 容差；None 表示跳过门控。

    Returns:
        tuple[float, float, float, float]: (p10_ms, p50_ms, p90_ms, tok_s_p50)。

    Raises:
        CorrectnessError: 输出与参考不一致时（由上层捕获并 SKIP 该 case）。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    state = make_state(model, batch_size)
    tokens = torch.arange(batch_size * seq_len, dtype=torch.long, device=device).view(
        batch_size, seq_len
    )
    tokens = (tokens * 1103515245 + 12345) % 65536
    input_tokens = _prepare_tokens(model, tokens)

    if reference is not None and correctness_tol is not None:
        got = _fresh_logits(model, input_tokens, batch_size)
        ref_input = _prepare_tokens(reference, tokens)
        ref = _fresh_logits(reference, ref_input, batch_size)
        _check_correctness(got, ref, type(model).__name__, correctness_tol)

    for _ in range(warmup):
        _ = _eager_dispatch(model, input_tokens, state)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = _eager_dispatch(model, input_tokens, state)
            end.record()
            torch.cuda.synchronize(device=device)
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            _ = _eager_dispatch(model, input_tokens, state)
            times.append((time.perf_counter() - t0) * 1000.0)

    p10 = percentile(times, 10)
    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    tok_s = batch_size * seq_len * 1000.0 / p50
    return p10, p50, p90, tok_s


def _print_row(label: str, B: int, T: int, iters: int, p10, p50, p90, tok_s) -> None:
    """打印一行 RESULT 与一行 csv。

    Args:
        label (str): 实现标签。
        B (int): 批大小。
        T (int): 序列长度。
        iters (int): 计时轮数。
        p10/p50/p90 (float): 分位延迟（毫秒）。
        tok_s (float): p50 对应吞吐（token/s）。
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


def build_fast_model(module_path: Path, model_path: str):
    """加载 fast 实现模块并构造 RWKV7（CUDA）。

    Args:
        module_path (Path): rwkv7_fast_v3a.py 的完整路径。
        model_path (str): 模型权重路径。

    Returns:
        fast 模块内的 RWKV7 实例。
    """
    module = load_fast_module(module_path)
    module.MODEL_PATH = model_path  # pyright: ignore[reportAttributeAccessIssue]
    module.load_extensions()
    return module.RWKV7()


def run_benchmark(args):
    """根据 --targets 与 --device 运行选定实现的计时。

    faster3a_2607 始终 CUDA；项目 target 运行在 --device。
    fla / FlashRWKV 先保留为占位 target，后续再接入。

    Args:
        args: argparse 解析结果。
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

    correctness_tol = args.correctness_tol

    for target in targets:
        model = None
        w: RWKV7Weight | None = None
        reference: RWKV7Model | None = None
        try:
            if target == "faster3a_2607":
                # Albatross is CUDA-only: skip (not error) when unavailable or when
                # the user explicitly requested --device cpu for the project targets.
                if rwkv_device.type != "cuda" or not torch.cuda.is_available():
                    print(
                        f"SKIP label=faster3a_2607 reason=cuda_only_device={rwkv_device.type}",
                        flush=True,
                    )
                    continue
                model = build_fast_model(
                    Path(args.fast_script) / "rwkv7_fast_v3a.py",
                    args.project_checkpoint,
                )
                device = torch.device("cuda")
                gate_dtype: torch.dtype | None = None
            elif target in BACKEND_FOR_TARGET:
                # One fresh weight per target, freed after the target's cases:
                # only ONE weight copy is resident in VRAM at any time (MX450
                # has 2GB and the correctness reference shares this same object).
                dtype = DTYPE_FOR_TARGET[target]
                w = RWKV7Weight(
                    str(args.project_checkpoint), device=rwkv_device, dtype=dtype
                )
                model_cls = make_rwkv7(
                    rwkv_device,
                    backend=BACKEND_FOR_TARGET[target],
                )
                model = model_cls(w, is_torch_compile=args.compile)
                device = rwkv_device
                gate_dtype = dtype
                if target == "tl-tuned":
                    # The tuned selector is device-name based; surface which
                    # variant was picked so a run is reproducible on paper.
                    print(
                        f"MODEL label={target} class={type(model_cls).__name__}",
                        flush=True,
                    )
            elif target in {"fla", "FlashRWKV"}:
                raise NotImplementedError(
                    f"target '{target}' is reserved but not implemented yet"
                )
            else:
                raise ValueError(f"unknown target: {target}")

            # Correctness gate (off by default): compare each target's output
            # against a pure-torch reference that SHARES the target's weight
            # object (no extra VRAM). pure-torch is self-consistent; third-party
            # faster3a_2607 is not gated.
            if args.correctness_check and target in GATED_TARGETS:
                assert w is not None and gate_dtype is not None
                ref_cls = make_rwkv7(rwkv_device, backend="torch")
                reference = ref_cls(w, is_torch_compile=False)  # type: ignore[call-arg]

            for B, T in parsed_cases:
                try:
                    p10, p50, p90, tok_s = bench_case(
                        model,
                        B,
                        T,
                        args.warmup,
                        args.iters,
                        device,
                        reference,
                        correctness_tol,
                    )
                except CorrectnessError as exc:
                    print(
                        f"SKIP label={target}({device.type}) B={B} T={T} "
                        f"reason=incorrect ({exc})",
                        flush=True,
                    )
                    continue
                except RuntimeError as exc:
                    # A single case OOM-ing must not abort the whole benchmark
                    # run; skip it and keep going so later (smaller) cases and
                    # other targets still get measured.
                    if "out of memory" not in str(exc).lower():
                        raise
                    print(
                        f"SKIP label={target}({device.type}) B={B} T={T} reason=oom",
                        flush=True,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                    continue
                _print_row(
                    f"{target}({device.type})", B, T, args.iters, p10, p50, p90, tok_s
                )
        finally:
            del model
            del reference
            if w is not None:
                del w
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


def main():
    """主入口：解析参数并运行 benchmark。"""
    default_vocab = str(REPO_ROOT / "asset" / "rwkv_vocab_v20230424.txt")
    parser = argparse.ArgumentParser(
        description="Benchmark multiple RWKV7 implementations"
    )
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
        default="faster3a_2607,tl-fp16,pure-torch",
        help=(
            "Comma/space separated targets: faster3a_2607, tl-fp16, tl-bf16, "
            "tl-tuned, pure-torch, fla, FlashRWKV. Defaults to the implemented "
            "targets."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Controls the project-target device: cpu | cuda. faster3a_2607 (CUDA-only) auto-skips on cpu.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Build the tilelang / pure-torch targets with is_torch_compile=True "
        "(decode goes through torch.compile + custom ops). Each distinct token "
        "count recompiles a fresh graph (minutes), so prefer eager for sweeps; "
        "this flag is for comparing compiled vs eager on a fixed case.",
    )
    parser.add_argument(
        "--cases",
        default="1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16",
    )
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Enable the pure-torch correctness gate (default off: keeps VRAM "
        "low, important on 2GB GPUs like MX450). SKIPs cases whose output does "
        "not match the same-dtype pure-torch reference.",
    )
    parser.add_argument(
        "--correctness-tol",
        type=float,
        default=16.0,
        help="Max-abs logit tolerance for the correctness gate (default 16.0).",
    )
    args = parser.parse_args()

    if not args.project_checkpoint:
        parser.error(
            "--project-checkpoint is required or RWKV_CHECKPOINT_PATH must be set"
        )
    parsed_targets = parse_targets(args.targets)
    if "faster3a_2607" in parsed_targets and not args.fast_script:
        parser.error("--fast-script is required or RWKV_FAST_SCRIPT_PATH must be set")
    if not args.vocab:
        parser.error("--vocab is required")

    run_benchmark(args)


if __name__ == "__main__":
    main()
