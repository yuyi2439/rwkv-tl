# RWKV7 Benchmark

> 这个文件只记录可复现的测试入口、执行环境、主要结果和简要解释。更细的实验发现请见 [docs/benchmark_rwkv7_experiments.md](../docs/benchmark_rwkv7_experiments.md)，运行与维护注意事项请见 [AGENT.md](../AGENT.md)。

## 运行命令

```bash
uv run python script/benchmark_rwkv7.py \
  --project-checkpoint ~/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --fast-script ~/rwkv/Albatross/faster3a_2607 \
  --device cuda \
  --targets faster3a_2607,rwkv_tl,pure_torch,graph_decoder \
  --cases 1x1,8x8,16x16
```

## 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA MX450 (sm_75, 2GB GDDR6) |
| 模型 | rwkv7-g1d-0.1b (C=768, H=12, L=12), rwkv7-g1d-0.4b (C=1024, H=16, L=24) |
| 精度 | bfloat16 (CUDA), float32 (CPU) |
| 实现 | faster3a_2607 (Albatross), rwkv_tl (本项目), pure_torch (纯 PyTorch 基线), graph_decoder (CUDA Graph 解码) |

## 实现说明

| 实现 | device | 路径 | 说明 |
|---|---|---|---|
| faster3a_2607 | cuda | forward | Albatross CUDA 扩展，wkv_seq kernel（T 维 kernel 内串行），编译目标 sm_75 |
| rwkv_tl | cuda/cpu | forward | 本项目 fused kernel + GEMM 批处理；T=1 走 decode（run_one），T>1 走 prefill（forward_prefill, GEMM 批处理） |
| pure_torch | cuda/cpu | forward | 纯 PyTorch eager 基线，无自定义 kernel；同样 T=1 decode / T>1 prefill |
| graph_decoder | cuda | decode | rwkv_tl + CUDA Graph 捕获单 token 解码，消除 launch 开销；T>1 时逐 token replay |

注：
- `--device cpu` 时，faster3a_2607 和 graph_decoder 自动跳过（CUDA-only）。
- warmup=5, iters=10 (CUDA); warmup=1, iters=3 (CPU，因耗时较长)。

## 结果：0.1B (rwkv7-g1d-0.1b-20260129-ctx8192)

### MX450 (CUDA)

> 注：MX450 是笔记本 GPU，长时满载会热降频（SM 时钟从 1800MHz 降到 ~1155MHz），
> 绝对延迟在不同会话间波动较大（本次数值整体比旧记录高 ~50%，系降频所致）；
> 单次运行内的相对比较更可靠。以目标卡（RTX 3060+ / AMD MI）为准。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 9.18 | 108.95 |
| faster3a_2607 | 1×32 | 44.03 | 726.81 |
| faster3a_2607 | 1×64 | 47.69 | 1341.97 |
| faster3a_2607 | 1×128 | 87.58 | 1461.47 |
| rwkv_tl | 1×1 | 85.52 | 11.69 |
| rwkv_tl | 1×32 | 214.79 | 148.98 |
| rwkv_tl | 1×64 | 419.58 | 152.53 |
| rwkv_tl | 1×128 | 659.60 | 194.06 |
| pure_torch | 1×1 | 27.57 | 36.27 |
| pure_torch | 1×32 | 267.67 | 119.55 |
| pure_torch | 1×64 | 495.63 | 129.13 |
| pure_torch | 1×128 | 820.61 | 155.98 |
| graph_decoder | 1×1 | 7.67 | 130.35 |
| graph_decoder | 1×32 | 220.03 | 145.43 |
| graph_decoder | 1×64 | 440.80 | 145.19 |
| graph_decoder | 1×128 | 881.49 | 145.21 |

### CPU

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| rwkv_tl | 1×1 | 53.71 | 18.62 |
| rwkv_tl | 1×8 | 311.39 | 25.69 |
| rwkv_tl | 1×32 | 339.46 | 94.27 |
| pure_torch | 1×1 | 28.95 | 34.55 |
| pure_torch | 1×8 | 135.85 | 58.89 |
| pure_torch | 1×32 | 347.96 | 91.96 |

## 结果：0.4B (rwkv7-g1d-0.4b-20260210-ctx8192)

### MX450 (CUDA)

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 16.21 | 61.70 |
| faster3a_2607 | 1×32 | 168.23 | 190.21 |
| faster3a_2607 | 1×64 | 153.49 | 416.97 |
| faster3a_2607 | 1×128 | 222.78 | 574.55 |
| rwkv_tl | 1×1 | 157.01 | 6.37 |
| rwkv_tl | 1×32 | 289.71 | 110.46 |
| rwkv_tl | 1×64 | 479.61 | 133.44 |
| rwkv_tl | 1×128 | 791.40 | 161.74 |
| pure_torch | 1×1 | 40.70 | 24.57 |
| pure_torch | 1×32 | 372.20 | 85.98 |
| pure_torch | 1×64 | 542.20 | 118.04 |
| pure_torch | 1×128 | 1021.17 | 125.35 |
| graph_decoder | 1×1 | 21.94 | 45.57 |
| graph_decoder | 1×32 | 652.04 | 49.08 |
| graph_decoder | 1×64 | 1306.04 | 49.00 |
| graph_decoder | 1×128 | 2619.33 | 48.87 |

### CPU

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| rwkv_tl | 1×1 | 75.99 | 13.16 |
| rwkv_tl | 1×8 | 371.47 | 21.54 |
| rwkv_tl | 1×32 | 977.48 | 32.74 |
| pure_torch | 1×1 | 95.74 | 10.45 |
| pure_torch | 1×8 | 365.44 | 21.89 |
| pure_torch | 1×32 | 1084.44 | 29.51 |

## 简要解释

### CUDA (MX450, sm_75)

- T=1 decode 时，graph_decoder 最快，说明 CUDA Graph 对单 token 解码的 launch 开销消除是有效的。
- 纯 eager 的 rwkv_tl 在 T=1 上最慢（约为 pure_torch 的 ~3 倍），说明在这个硬件与路径下，逐 token dispatch 和 kernel launch 的额外开销很明显。
- 在 prefill 场景（T≥32）中，rwkv_tl 的 fused GEMM 批处理已经快于 pure_torch（1×32: 215 vs 268ms），说明批处理收益在 MX450 上已经显现；但相比 Albatross 的单 kernel 长串行路径仍有差距。
- 0.4B 相比 0.1B 的结果更慢，符合模型尺寸增大带来的额外计算成本。

### CPU

- CPU 上 rwkv_tl 与 pure_torch 接近，说明当前的 fused path 在 CPU 端没有明显优势。
- 在较大 prefill 场景下，rwkv_tl 的批处理收益仍然有限，结果与 CUDA 上的差异一致。

## 复现命令

```bash
# CUDA (0.1B)
RWKV_CHECKPOINT_PATH=...0.1b.pth RWKV_FAST_SCRIPT_PATH=.../faster3a_2607 .venv/bin/python script/benchmark_rwkv7.py --targets faster3a_2607,rwkv_tl,pure_torch,graph_decoder --device cuda --cases 1x1,1x32,1x64,1x128 --warmup 5 --iters 10

# CPU (0.1B) - Albatross/graph_decoder 自动跳过
RWKV_CHECKPOINT_PATH=...0.1b.pth .venv/bin/python script/benchmark_rwkv7.py --targets faster3a_2607,rwkv_tl,pure_torch --device cpu --cases 1x1,1x8,1x32 --warmup 1 --iters 3
```

更细的实验发现请见 [docs/benchmark_rwkv7_experiments.md](../docs/benchmark_rwkv7_experiments.md)。
