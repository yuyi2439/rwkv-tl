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
| GPU (目标卡) | NVIDIA RTX 3060 (sm_86, 12GB GDDR6) |
| GPU (旧参考) | NVIDIA MX450 (sm_75, 2GB GDDR6) |
| 模型 | rwkv7-g1d-0.1b (C=768, H=12, L=12), rwkv7-g1d-0.4b (C=1024, H=16, L=24) |
| 精度 | bfloat16 (CUDA), float32 (CPU) |
| 实现 | faster3a_2607 (Albatross), rwkv_tl (本项目), pure_torch (纯 PyTorch 基线), graph_decoder (CUDA Graph 解码) |

## 实现说明

| 实现 | device | 路径 | 说明 |
|---|---|---|---|
| faster3a_2607 | cuda | forward | Albatross CUDA 扩展，wkv_seq kernel（T 维 kernel 内串行），编译目标 sm_75 |
| rwkv_tl | cuda/cpu | forward | 本项目 fused kernel + 单 kernel prefill（fused_dplr_T：T 维串行递推在 kernel 内，一次 launch 交付整个序列）；T=1 走 decode（decode），T>1 走 prefill |
| pure_torch | cuda/cpu | forward | 纯 PyTorch eager 基线，无自定义 kernel；同样 T=1 decode / T>1 prefill |
| graph_decoder | cuda | decode | rwkv_tl + CUDA Graph 捕获单 token 解码，消除 launch 开销；T>1 时逐 token replay |

注：
- `--device cpu` 时，faster3a_2607 和 graph_decoder 自动跳过（CUDA-only）。
- warmup=5, iters=10 (CUDA); warmup=1, iters=3 (CPU，因耗时较长)。
- 正确性门控默认开启：每个 case 计时前先把输出与 pure_torch 参考对比（argmax 一致且 max_abs ≤ 16），不一致则该 case 输出 `SKIP reason=incorrect` 且不报延迟。用 `--no-correctness-check` 关闭。
- **MX450 2GB 显存注意**：0.4B 模型 + pure_torch 参考模型合计 ~2.4GB 超出显存，正确性门控会触发内存压力导致 0.4B rwkv_tl 延迟虚高 10x。0.4B MX450 数据使用 `--no-correctness-check` 采集。

## 结果：0.1B (rwkv7-g1d-0.1b-20260129-ctx8192)

### RTX 3060 (CUDA, sm_86, 目标卡)

warmup=10, iters=20。rwkv_tl / pure_torch 走 eager 路径（benchmark 不触发 torch.compile 重编译）。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 4.32 | 231.50 |
| faster3a_2607 | 1×8 | 6.23 | 1284.88 |
| faster3a_2607 | 1×32 | 7.49 | 4274.21 |
| faster3a_2607 | 8×8 | 7.64 | 8380.39 |
| faster3a_2607 | 16×16 | 7.64 | 33526.74 |
| rwkv_tl | 1×1 | 40.48 | 24.70 |
| rwkv_tl | 1×8 | 40.48 | 197.63 |
| rwkv_tl | 1×32 | 118.12 | 270.92 |
| rwkv_tl | 8×8 | 206.94 | 309.26 |
| rwkv_tl | 16×16 | 758.45 | 337.53 |
| pure_torch | 1×1 | 13.74 | 72.77 |
| pure_torch | 1×8 | 29.59 | 270.37 |
| pure_torch | 1×32 | 98.20 | 325.87 |
| pure_torch | 8×8 | 171.77 | 372.58 |
| pure_torch | 16×16 | 609.80 | 419.81 |
| graph_decoder | 1×1 | 2.11 | 473.14 |
| graph_decoder | 1×8 | 14.76 | 541.88 |
| graph_decoder | 1×32 | 58.78 | 544.42 |

### MX450 (CUDA, sm_75, 旧参考)

> 注：MX450 是笔记本 GPU，长时满载会热降频（SM 时钟从 1800MHz 降到 ~1155MHz），
> 绝对延迟在不同会话间波动较大（本次数值整体比旧记录高 ~50%，系降频所致）；
> 单次运行内的相对比较更可靠。以目标卡（RTX 3060+ / AMD MI）为准。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 9.80 | 102.02 |
| faster3a_2607 | 1×32 | 43.93 | 728.39 |
| faster3a_2607 | 1×64 | 46.93 | 1363.64 |
| faster3a_2607 | 1×128 | 87.84 | 1457.13 |
| rwkv_tl | 1×1 | 15.94 | 62.72 |
| rwkv_tl | 1×32 | 24.71 | 1295.03 |
| rwkv_tl | 1×64 | 27.74 | 2306.85 |
| rwkv_tl | 1×128 | 37.56 | 3407.73 |
| pure_torch | 1×1 | 23.01 | 43.46 |
| pure_torch | 1×32 | 248.25 | 128.90 |
| pure_torch | 1×64 | 410.57 | 155.88 |
| pure_torch | 1×128 | 753.04 | 169.98 |
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

### RTX 3060 (CUDA, sm_86, 目标卡)

warmup=10, iters=20。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 6.50 | 153.94 |
| faster3a_2607 | 1×8 | 13.78 | 580.62 |
| faster3a_2607 | 1×32 | 16.25 | 1969.52 |
| faster3a_2607 | 8×8 | 16.28 | 3931.92 |
| faster3a_2607 | 16×16 | 16.84 | 15198.95 |
| rwkv_tl | 1×1 | 85.09 | 11.75 |
| rwkv_tl | 1×8 | 69.66 | 114.84 |
| rwkv_tl | 1×32 | 179.03 | 178.74 |
| rwkv_tl | 8×8 | 307.36 | 208.23 |
| rwkv_tl | 16×16 | 1128.89 | 226.77 |
| pure_torch | 1×1 | 8.69 | 115.01 |
| pure_torch | 1×8 | 72.56 | 110.25 |
| pure_torch | 1×32 | 175.51 | 182.33 |
| pure_torch | 8×8 | 347.61 | 184.11 |
| pure_torch | 16×16 | 1294.50 | 197.76 |
| graph_decoder | 1×1 | 4.60 | 217.18 |
| graph_decoder | 1×8 | 36.24 | 220.72 |
| graph_decoder | 1×32 | 144.16 | 221.97 |

### MX450 (CUDA, sm_75, 旧参考)

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 14.06 | 71.14 |
| faster3a_2607 | 1×32 | 165.61 | 193.23 |
| faster3a_2607 | 1×64 | 150.45 | 425.39 |
| faster3a_2607 | 1×128 | 212.75 | 601.63 |
| rwkv_tl | 1×1 | 21.35 | 46.84 |
| rwkv_tl | 1×32 | 56.76 | 563.74 |
| rwkv_tl | 1×64 | 67.54 | 947.56 |
| rwkv_tl | 1×128 | 110.98 | 1153.36 |
| pure_torch | 1×32 | 346.67 | 92.31 |
| pure_torch | 1×64 | 638.33 | 100.26 |
| pure_torch | 1×128 | 1328.24 | 96.37 |
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

### CUDA (RTX 3060, sm_86, 目标卡)

- T=1 decode 时 graph_decoder 最快（0.1B 2.11ms、0.4B 4.60ms），CUDA Graph 消除 launch 开销的效果在 sm_86 上依旧成立。
- eager rwkv_tl 的 T=1 仍明显慢于 pure_torch（0.1B 40.5 vs 13.7ms；0.4B 85.1 vs 8.7ms），说明 fused kernel 的逐 token dispatch 开销在小模型上依旧占主导。
- prefill（T≥32）rwkv_tl 与 pure_torch 接近（0.1B 1×32 118 vs 98ms；0.4B 1×32 179 vs 176ms），fused GEMM 批处理与纯 PyTorch 的 batched 路径基本打平，收益不如 MX450 上明显。
- faster3a_2607 在所有 case 上大幅领先：prefill 已是 kernel 内串行的 T 维处理，B×T 增大几乎不影响单次延迟（0.4B 1×32 到 16×16 都约 16ms）。
- 编译 prefill 的结论：torch.compile 后 0.1B prefill 快 1.11-1.43x（T=8~256），但每个不同 T 都会重编译一张图（T=256 约 12 分钟，GPU 空闲），收益不抵成本，故 `prefill` 保持 eager。详见 docs/benchmark_rwkv7_experiments.md。

### CUDA (MX450, sm_75)

- T=1 decode 时，graph_decoder 最快，说明 CUDA Graph 对单 token 解码的 launch 开销消除是有效的。
- **单 kernel prefill（fused_dplr_T）**：state 串行递推在 kernel 内、一次 launch 交付整个序列 + fp32io16 state。prefill 大幅提速并**反超 faster3a_2607**：
  - 0.1B：1×32 24.7ms（1295 tok/s）vs faster3a 43.9ms（728 tok/s），快 **1.78x**；1×128 37.6ms（3408 tok/s）vs faster3a 87.8ms（1457 tok/s），快 **2.34x**。
  - 0.4B：1×32 56.8ms（564 tok/s）vs faster3a 165.6ms（193 tok/s），快 **2.92x**；1×128 111.0ms（1153 tok/s）vs faster3a 212.8ms（602 tok/s），快 **1.92x**。
- 对比 pure_torch：0.1B 1×128 快 20x（37.6 vs 753ms），0.4B 1×128 快 12x（111 vs 1328ms）。
- T=1 decode 仍慢于 faster3a_2607（0.1B 15.9 vs 9.8ms，0.4B 21.4 vs 14.1ms），因 fused kernel 逐 token dispatch 开销；graph_decoder（CUDA Graph）可弥补此差距。
- faster3a_2607 在 0.4B 上 T=64（150ms）反比 T=32（166ms）快，因其 chunk kernel 对不同序列长度有不同性能特征。
- **MX450 2GB 显存限制**：0.4B 正确性门控会同时加载 pure_torch 参考模型（合计 ~2.4GB > 2GB），导致 rwkv_tl 延迟虚高 10x（540ms vs 实际 56ms）。0.4B 数据用 `--no-correctness-check` 采集。

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
