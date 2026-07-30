#!/usr/bin/env python3
"""使用 torch.profiler 分析 rwkv_tl.RWKV7 前向性能。

对项目 RWKV7 实现的 forward 进行 profiling，产出：
1. 运行环境信息（python / torch / cuda / gpu）
2. 算子级 key_averages 表格（分别按 self 时间与 self 内存排序）
3. Chrome Trace JSON 文件（可用 chrome://tracing 打开）
4. 单 token 前向延迟统计（均值/中位数/最小/最大）

权重的 device 迁移沿用 compare_rwkv7_speed.py 的临时文件方案：
将原始 .pth 的 tensor 迁移到目标 device 后另存临时文件，
再交给 RWKV7.__init__ 加载，避免侵入库代码。

Args via argparse:
    --checkpoint: 模型权重路径
    --vocab: 词表文件路径
    --device: cpu | cuda
    --seq-len: profile 使用的 token 序列长度
    --active: profiler active 阶段重复次数
    --trace-out: Chrome Trace JSON 输出路径

Callers:
    - 手动运行: `python script/profile_rwkv7.py --device cuda`
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwkv_tl import RWKV7  # noqa: E402


def build_model(checkpoint: Path, vocab: Path, device: torch.device) -> RWKV7:
    """加载权重并迁移到目标 device，返回构造好的 RWKV7 实例。

    RWKV7.__init__ 内部会再次 torch.load，因此采用“迁移后另存临时文件”
    的方式把 tensor 放到目标 device，再传路径给 RWKV7。

    Args:
        checkpoint (Path): 原始 .pth 权重路径。
        vocab (Path): 词表文件路径。
        device (torch.device): 目标设备。

    Returns:
        RWKV7: 权重已位于 device 上的模型实例。

    Callers:
        - `profile_rwkv7.py:main`: 主入口调用。
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    moved = {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in ckpt.items()
    }
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(moved, tmp.name)
        tmp_path = Path(tmp.name)
    try:
        model = RWKV7(str(tmp_path), str(vocab))
    finally:
        tmp_path.unlink(missing_ok=True)
    return model


def state_to_device(state, device: torch.device):
    """把 zero_state 产生的 state 中所有 tensor 迁移到目标 device。

    RWKV7.zero_state 返回的 tensor 默认在 CPU，与 CUDA 权重混算会报错，
    因此在每次重置 state 后调用本函数做迁移。

    Args:
        state: RWKV7.zero_state() 的返回值（嵌套 list[dict]）。
        device (torch.device): 目标设备。

    Returns:
        list: 迁移后的 state（原地修改并返回）。

    Callers:
        - `profile_rwkv7.py:main`: 每次重置 state 后调用。
    """
    for layer_state in state:
        for slot in layer_state:
            for k, v in slot.items():
                slot[k] = v.to(device=device)
    return state


def print_env(device: torch.device) -> None:
    """打印运行环境信息。

    Args:
        device (torch.device): 实际使用的设备。

    Callers:
        - `profile_rwkv7.py:main`: 启动时调用。
    """
    print(f"python : {sys.version.split()[0]}")
    print(f"torch  : {torch.__version__}")
    print(f"device : {device}  (cuda_available={torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"gpu    : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"gpu_mem: free={free/1024**2:.1f}MiB total={total/1024**2:.1f}MiB")


def print_tm_cm(prof, device: torch.device) -> None:
    """从 profiler 结果中提取 TM/CM record_function 分项统计。

    依赖 RWKV7.run_one 内的 torch.profiler.record_function("TM")/("CM") 标签。
    输出 time_mixing 与 channel_mixing 的总耗时、调用次数与平均耗时，便于
    直接对比二者开销。

    Args:
        prof: torch.profiler.profile 上下文返回的对象。
        device (torch.device): 目标设备，决定用 cuda_time_total 还是 cpu_time_total。

    Callers:
        - `profile_rwkv7.py:profile_forward`: 表格输出后调用。
    """
    field = "device_time_total" if device.type == "cuda" else "cpu_time_total"
    totals = {"TM": 0.0, "CM": 0.0}
    counts = {"TM": 0, "CM": 0}
    for ev in prof.key_averages():
        if ev.key in totals:
            totals[ev.key] += getattr(ev, field)
            counts[ev.key] += ev.count
    print("\n=== TM / CM breakdown (record_function) ===")
    for k in ("TM", "CM"):
        c = counts[k]
        if c == 0:
            print(f"  {k}: (not recorded)")
            continue
        total_ms = totals[k] / 1000.0  # us -> ms
        print(f"  {k}: total={total_ms:.3f}ms  calls={c}  avg={total_ms / c:.4f}ms")


def profile_forward(
    model: RWKV7,
    tokens: list[int],
    device: torch.device,
    active: int,
    trace_out: Path,
) -> None:
    """用 torch.profiler 分析 forward 的算子级开销并导出 Chrome Trace。

    采用 schedule(wait=1, warmup=1, active=N)，每次 step 对一个完整
    token 序列做一次 forward（内部逐 token 推进 RNN 状态）。

    Args:
        model (RWKV7): 已构造的模型。
        tokens (list[int]): 输入 token 序列。
        device (torch.device): 目标设备。
        active (int): active 阶段重复次数。
        trace_out (Path): Chrome Trace JSON 输出路径。

    Callers:
        - `profile_rwkv7.py:main`: 主入口调用。
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_steps = 1 + 1 + active  # wait + warmup + active
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=active, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            S = state_to_device(model.zero_state(), device)
            model.forward(tokens, S)
            prof.step()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(f"\n=== key_averages sorted by {sort_key} ===")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))

    print("\n=== key_averages sorted by self_cpu_memory_usage ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    print_tm_cm(prof, device)

    prof.export_chrome_trace(str(trace_out))
    print(f"\nchrome trace -> {trace_out}")


def bench_per_token(
    model: RWKV7,
    tokens: list[int],
    device: torch.device,
    rounds: int,
) -> None:
    """逐 token 计时，给出单 token 前向延迟统计。

    每个 round 重置 state 后推进整个序列，逐 token 记录端到端延迟
    （CUDA 设备在前后做 synchronize）。

    Args:
        model (RWKV7): 已构造的模型。
        tokens (list[int]): 输入 token 序列。
        device (torch.device): 目标设备。
        rounds (int): 重复轮数。

    Callers:
        - `profile_rwkv7.py:main`: 主入口调用。
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(rounds):
        S = state_to_device(model.zero_state(), device)
        for t in tokens:
            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.forward([t], S)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
            else:
                t0 = time.perf_counter()
                model.forward([t], S)
                latencies.append((time.perf_counter() - t0) * 1000.0)
    n = len(latencies)
    print(f"\nper-token latency over {n} samples (seq_len={len(tokens)}, rounds={rounds}):")
    print(
        f"  mean={statistics.mean(latencies):.3f}ms  "
        f"median={statistics.median(latencies):.3f}ms  "
        f"min={min(latencies):.3f}ms  max={max(latencies):.3f}ms"
    )
    print(f"  est full-seq fwd ≈ {statistics.median(latencies) * len(tokens):.2f}ms")


def main() -> None:
    """主入口：解析参数、打印环境、构造模型、运行 profiler 与逐 token 计时。

    Callers:
        - 命令行: `python script/profile_rwkv7.py`
    """
    default_ckpt = "/home/yuyi2439/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    default_vocab = str(REPO_ROOT / "asset" / "rwkv_vocab_v20230424.txt")
    parser = argparse.ArgumentParser(description="Profile rwkv_tl.RWKV7 forward")
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--vocab", default=default_vocab)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument(
        "--trace-out", default=str(REPO_ROOT / "profile_trace.json")
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="对每层 TM/CM 闭包套 torch.compile，测编译前后差异",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile 模式",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print_env(device)
    print(f"compile: {args.compile}  mode={args.compile_mode}")

    ckpt_path = Path(args.checkpoint)
    vocab_path = Path(args.vocab)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab not found: {vocab_path}")

    model = build_model(ckpt_path, vocab_path, device)
    # 打印首层部分权重 dtype，确认精度
    sample_w = model.W["blocks.0.att.receptance.weight"]
    print(f"weight dtype sample: {sample_w.dtype}  (device={sample_w.device})")

    if args.compile:
        # 对每层 time_mixing / channel_mixing 闭包做 torch.compile。
        # 闭包内部会原地更新 state dict（state["x"]/state["rnn"]），
        # torch.compile 默认模式可处理此类副作用；reduce-overhead 依赖
        # cudaGraph，对逐 token 串行 RNN 通常不适用，故默认用 default。
        mode = None if args.compile_mode == "default" else args.compile_mode
        model.TM = tuple(torch.compile(tm, mode=mode) for tm in model.TM)
        model.CM = tuple(torch.compile(cm, mode=mode) for cm in model.CM)

    # 固定可复现的伪随机 token 序列
    tokens = [(i * 1103515245 + 12345) % 65536 for i in range(args.seq_len)]

    # 预热：让 CUDA 内核 / 内存分配稳定；compile 模式下首轮触发编译
    warmup_rounds = 5 if args.compile else 2
    for _ in range(warmup_rounds):
        S = state_to_device(model.zero_state(), device)
        model.forward(tokens, S)
    if device.type == "cuda":
        torch.cuda.synchronize()

    profile_forward(model, tokens, device, args.active, Path(args.trace_out))
    bench_per_token(model, tokens, device, rounds=args.active)


if __name__ == "__main__":
    main()
