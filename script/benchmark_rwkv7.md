# RWKV7 Benchmark

> 这个文件只记录可复现的测试入口、执行环境、主要结果和简要解释。更细的实验发现请见 [docs/benchmarks/rtx3060.md](../docs/benchmarks/rtx3060.md)，运行与维护注意事项请见 [AGENT.md](../AGENT.md)。

## 运行命令

```bash
uv run python script/benchmark_rwkv7.py \
  --project-checkpoint ~/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --fast-script ~/rwkv/Albatross/faster3a_2607 \
  --device cuda \
  --targets faster3a_2607,tl-fp16,pure-torch \
  --cases 1x1,8x8,16x16
```

## 环境

| 项目 | 值 |
|---|---|
| GPU (目标卡) | NVIDIA RTX 3060 (sm_86, 12GB GDDR6) |
| GPU (旧参考) | NVIDIA MX450 (sm_75, 2GB GDDR6) |
| 模型 | rwkv7-g1d-0.1b (C=768, H=12, L=12), rwkv7-g1d-0.4b (C=1024, H=16, L=24) |
| 精度 | float16 (CUDA), float32 (CPU) |
| 实现 | faster3a_2607 (Albatross), tl-fp16 (本项目 fp16), tl-bf16 (本项目 bf16), tl-mx450 (sm_75 变体, CUDA Graph decode), pure-torch (纯 PyTorch 基线) |

## 实现说明

| 实现 | device | 路径 | 说明 |
|---|---|---|---|
| faster3a_2607 | cuda | forward | Albatross CUDA 扩展，wkv_seq kernel（T 维 kernel 内串行），sm_75 适配版来自 [yuyi2439/Albatross `support/sm75`](https://github.com/yuyi2439/Albatross/tree/support/sm75) |
| tl-fp16 | cuda/cpu | forward | 本项目 fused kernel + 单 kernel prefill（fused_dplr_T：T 维串行递推在 kernel 内，一次 launch 交付整个序列）；fp16 权重；T=1 走 decode（decode），T>1 走 prefill |
| tl-bf16 | cuda/cpu | forward | 同 tl-fp16，但用 checkpoint 原始 bf16 权重（bf16 kernel 绑定），参考/实验用 |
| tl-mx450 | cuda | forward | fp16 变体，sm_75 下 batch prefill GEMM 走 fp32 快路径（Turing fp16 cuBLAS 小 shape 病态慢）+ T≤16 rkv 走 tilelang fp16；decode 走 CUDA Graph |
| pure-torch | cuda/cpu | forward | 纯 PyTorch eager 基线，无自定义 kernel；同样 T=1 decode / T>1 prefill |

注：
- `--device cpu` 时，faster3a_2607 自动跳过（CUDA-only）。
- graph_decoder benchmark target 已移除：CUDA Graph 现由 `demo.cuda_graph.CUDAGraph` 通用包装器提供，`tl-mx450`/`tl-rtx3060`/`tl-tuned` 通过 `make_rwkv7(use_graph=True)`（默认）自动叠加 decode + 小 T prefill 的 graph。
- warmup=5, iters=10 (CUDA); warmup=1, iters=3 (CPU，因耗时较长)。
- 正确性门控**默认关闭**（`--correctness-check` 开启）：每个 case 计时前先把输出与同 dtype 的 pure_torch 参考对比（argmax 一致且 max_abs ≤ 16），不一致则该 case 输出 `SKIP reason=incorrect` 且不报延迟。默认关是为了省显存（参考模型共享 target 权重对象，但多 target 混跑仍可能压 2GB 卡）。
- **权重按 target 加载/释放**：每个 target 独立 `RWKV7Weight`，跑完即删（`del` + `gc.collect()` + `empty_cache()`），同进程同时只有一份权重在显存。
- **MX450 2GB 显存注意**：0.4B 模型 + pure_torch 参考模型合计 ~2.4GB 超出显存，正确性门控会触发内存压力导致 0.4B rwkv_tl 延迟虚高 10x。0.4B MX450 数据不要开 `--correctness-check` 采集。
- **同一进程跑多个 target 也会压显存**：fp16 + bf16 权重与参考模型同驻（0.1B 也会 ~1.6GB+），MX450 的 fp32 权重副本叠加后接近 2GB，延迟虚高数倍。MX450 数据建议单独跑该 target（实测 T=128 73ms，混跑 1029ms）。

## MX450 特调 vs 适配 sm75 的 faster3a_2607

> 对比对象：本项目 `tl-mx450`（fp32 prefill GEMM + T≤16 tilelang fp16 rkv + CUDA Graph decode）
> 与 [yuyi2439/Albatross](https://github.com/yuyi2439/Albatross) `support/sm75` 分支适配的
> faster3a_2607。0.1B / MX450，warmup=10, iters=20，单会话（仅两 target）。

| T | faster3a_2607 (sm75 适配) | tl-mx450 |
|---|---|---|
| 1 | 9.6ms（波动大） | **8.3ms（稳定）** |
| 2 | 11.4ms | **11.5ms（持平）** |
| 4 | **10.8ms** | 11.7ms |
| 8 | 23.6ms | **13.0ms** |
| 16 | 34.0ms | **19.5ms** |
| 32 | 43.9ms | **15.8ms** |
| 64 | 47.0ms | **22.5ms** |
| 128 | 88.6ms | **43.4ms** |

结论：
- **T=1 decode：mx450 稳定 8.3ms**（CUDA Graph 消除 launch 开销），faster3a 波动到 20ms+。
- **小 T prefill（T=2/4）持平，T≥8 全面反超**：2026-08-04 起小 T prefill 也走 CUDA Graph
  （launch 数恒定 ~2175 与 T 无关，T=4 从 33 → 11.7ms）。
- **T=128 快 2.04x（43.4 vs 88.6ms）**：GEMM 权重 `.T` 后补 `.contiguous()`（非连续
  cuBLAS 操作数在 Turing 慢 ~2.7x），70.6 → 43.4ms（-39%）。

原因分析（kernel 级剖析）见 [docs/benchmarks/mx450_sm75.md](../docs/benchmarks/mx450_sm75.md)。

## 结果：0.1B (rwkv7-g1d-0.1b-20260129-ctx8192)

### RTX 3060 (CUDA, sm_86, 目标卡)

warmup=10, iters=20，正确性门控全过。`tl-rtx3060`/`tl-mx450` 为 CUDA-Graph 加速变体
（decode + T≤64 per-T prefill graph），`tl-fp16` 为 eager fp16 基线。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 5.16 | 193.75 |
| faster3a_2607 | 1×8 | 6.61 | 1211.06 |
| faster3a_2607 | 1×32 | 8.89 | 3600.85 |
| faster3a_2607 | 1×64 | 8.36 | 7651.88 |
| faster3a_2607 | 1×128 | 7.60 | 16832.75 |
| faster3a_2607 | 8×8 | 7.95 | 8046.35 |
| faster3a_2607 | 16×16 | 8.37 | 30590.69 |
| tl-fp16 | 1×1 | 2.35 | 425.62 |
| tl-fp16 | 1×8 | 3.09 | 2585.69 |
| tl-fp16 | 1×32 | 3.37 | 9507.60 |
| tl-fp16 | 1×64 | 3.88 | 16482.00 |
| tl-fp16 | 1×128 | 5.38 | 23777.82 |
| tl-fp16 | 8×8 | 4.06 | 15745.92 |
| tl-fp16 | 16×16 | 17.15 | 14928.88 |
| tl-bf16 | 1×1 | 10.58 | 94.51 |
| tl-bf16 | 1×8 | 17.20 | 465.03 |
| tl-bf16 | 1×32 | 17.52 | 1827.01 |
| tl-bf16 | 1×64 | 16.60 | 3854.96 |
| tl-bf16 | 1×128 | 22.17 | 5774.38 |
| tl-bf16 | 8×8 | 18.44 | 3469.91 |
| tl-bf16 | 16×16 | 21.23 | 12056.62 |
| pure-torch | 1×1 | 4.93 | 202.93 |
| pure-torch | 1×8 | 5.77 | 1386.25 |
| pure-torch | 1×32 | 14.83 | 2158.07 |
| pure-torch | 1×64 | 28.39 | 2254.65 |
| pure-torch | 1×128 | 506.87 | 252.53 |
| pure-torch | 8×8 | 27.23 | 2350.41 |
| pure-torch | 16×16 | 1084.18 | 236.12 |

### MX450 (CUDA, sm_75, 旧参考)

> 注：MX450 是笔记本 GPU，长时满载会热降频（SM 时钟从 1800MHz 降到 ~1155MHz），
> 绝对延迟在不同会话间波动较大；单次运行内的相对比较更可靠。以目标卡（RTX 3060+ / AMD MI）为准。
>
> **fp16 迁移对 sm_75 的影响**：c2c4283 起 prefill 的批量 GEMM 从 bf16（cuBLAS magma
> fp32 模拟）改为 fp16。Turing 的 cuBLAS fp16 tensor-core 内核对 `[T,C]@[C,C]`（T=32..128）
> 病态慢（fp16 bmm ~1.3ms vs fp32 ~0.16ms，4-8x），导致 MX450 prefill 较旧记录 ~1.9x 变慢
> （46.4 vs 24.7ms @ T=32）。已按设备拆分模型类：`demo.rwkv7_fp16.RWKV7FP16`（全 fp16，sm_80+）
> 与 `demo.rwkv7_bf16.RWKV7BF16`、`demo.tuned.rwkv7_mx450.RWKV7MX450`（sm_75：decode 同 fp16，batch GEMM 走 fp32 快路径），
> `demo.make_rwkv7` 按 arch 自动选择。**2026-08-04 实测 tl-bf16 是 MX450 prefill 最快的变体**：
> T=8 20.5 vs tl-fp16 45.1ms，T=128 39.8 vs tl-fp16 92.2ms——bf16 的 tilelang kernel 在 Turing 走
> fp32 模拟路径，绕开了病态的 fp16 cuBLAS GEMM。
>
> **sm_75 fp16 GEMM（m16n8k8）已实现**：`rwkv_tl.kernels.gemm` 为 sm_75 的 fp16 加了
> T 特化 tilelang kernel（16×32×32/3 级流水，MX450 autotune），按允许长度集
> （1..16 精确 + 32..16384 幂）二分选择最小覆盖长度、pad 输入后切回。实测比病态
> cuBLAS fp16 bmm 快 4-6x（T=32 0.24 vs 0.79ms，T=128 0.83 vs 1.58ms），但**每个
> 不同长度首次调用编译一次**（~8s on MX450）。bf16 无 sm_75 MMA，保持 bmm；sm_80+
> 保持动态 kernel 不变。MX450 的 fp32 路径不受影响（dtype 检查兜底）。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 9.91 | 100.89 |
| faster3a_2607 | 1×8 | 24.16 | 331.07 |
| faster3a_2607 | 1×32 | 43.96 | 727.89 |
| faster3a_2607 | 1×64 | 46.55 | 1374.87 |
| faster3a_2607 | 1×128 | 87.67 | 1460.03 |
| tl-fp16 | 1×1 | 15.38 | 65.03 |
| tl-fp16 | 1×8 | 45.14 | 177.23 |
| tl-fp16 | 1×32 | 47.14 | 678.86 |
| tl-fp16 | 1×64 | 49.55 | 1291.58 |
| tl-fp16 | 1×128 | 92.18 | 1388.51 |
| tl-bf16 | 1×1 | 18.29 | 54.68 |
| tl-bf16 | 1×8 | 20.53 | 389.72 |
| tl-bf16 | 1×32 | 33.19 | 964.18 |
| tl-bf16 | 1×64 | 32.56 | 1965.86 |
| tl-bf16 | 1×128 | 39.79 | 3217.01 |
| pure-torch | 1×1 | 34.18 | 29.26 |
| pure-torch | 1×8 | 88.12 | 90.79 |
| pure-torch | 1×32 | 288.80 | 110.80 |
| pure-torch | 1×64 | 598.10 | 107.01 |
| pure-torch | 1×128 | 1332.23 | 96.08 |

> 注：本次运行（2026-08-04，warmup=5, iters=10，`tl-fp16,tl-bf16,pure-torch,faster3a_2607`，
> 正确性门控全过）未重新测 graph_decoder；0.1B MX450 graph_decoder 历史数据：T=1 7.96ms / T=32
> 218.73ms / T=128 882.25ms。

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

warmup=10, iters=20，正确性门控全过。`tl-fp16`/`pure-torch` 为 CUDA-Graph 包装
（decode + T≤64 prefill graph）。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 8.15 | 122.65 |
| faster3a_2607 | 1×8 | 13.73 | 582.85 |
| faster3a_2607 | 1×32 | 15.93 | 2008.87 |
| faster3a_2607 | 1×64 | 15.61 | 4099.97 |
| faster3a_2607 | 1×128 | 15.00 | 8538.93 |
| faster3a_2607 | 8×8 | 17.83 | 3589.68 |
| faster3a_2607 | 16×16 | 17.29 | 14808.03 |
| tl-fp16 | 1×1 | 5.21 | 191.78 |
| tl-fp16 | 1×8 | 7.49 | 1068.05 |
| tl-fp16 | 1×32 | 8.73 | 3664.66 |
| tl-fp16 | 1×64 | 10.46 | 6118.52 |
| tl-fp16 | 1×128 | 37.63 | 3401.87 |
| tl-fp16 | 8×8 | 10.41 | 6149.46 |
| tl-fp16 | 16×16 | 37.03 | 6913.27 |
| tl-bf16 | 1×1 | 24.98 | 40.03 |
| tl-bf16 | 1×8 | 37.67 | 212.37 |
| tl-bf16 | 1×32 | 37.71 | 848.51 |
| tl-bf16 | 1×64 | 35.82 | 1786.59 |
| tl-bf16 | 1×128 | 35.88 | 3567.88 |
| tl-bf16 | 8×8 | 37.17 | 1722.01 |
| tl-bf16 | 16×16 | 46.39 | 5518.27 |
| pure-torch | 1×1 | 8.16 | 122.51 |
| pure-torch | 1×8 | 14.17 | 564.49 |
| pure-torch | 1×32 | 33.74 | 948.38 |
| pure-torch | 1×64 | 59.22 | 1080.72 |
| pure-torch | 1×128 | 1221.29 | 104.81 |
| pure-torch | 8×8 | 61.13 | 1046.94 |
| pure-torch | 16×16 | 2337.45 | 109.52 |
| rwkv_tl | 1×64 | 31.38 | 2039.69 |
| rwkv_tl | 1×128 | 33.67 | 3801.59 |
| rwkv_tl | 8×8 | 32.74 | 1954.93 |
| rwkv_tl | 16×16 | 33.79 | 7577.32 |
| pure_torch | 1×1 | 32.59 | 30.68 |
| pure_torch | 1×8 | 87.14 | 91.81 |
| pure_torch | 1×32 | 289.14 | 110.67 |
| pure_torch | 1×64 | 503.39 | 127.14 |
| pure_torch | 1×128 | 1001.30 | 127.83 |
| pure_torch | 8×8 | 552.79 | 115.78 |
| pure_torch | 16×16 | 2000.69 | 127.96 |
| graph_decoder | 1×1 | 4.11 | 243.23 |
| graph_decoder | 1×8 | 31.69 | 252.41 |
| graph_decoder | 1×32 | 127.47 | 251.05 |
| graph_decoder | 1×64 | 254.74 | 251.23 |
| graph_decoder | 1×128 | 509.36 | 251.29 |

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

- **CUDA-Graph 加速变体全面领先**：`tl-rtx3060`（fp16 GEMM + CUDA-Graph decode + per-T prefill graph T≤64）在 T=1..64 全面超过 faster3a_2607（1×1 2.16 vs 4.40ms，1×8 2.86 vs 6.41ms，1×32 3.41 vs 8.16ms，1×64 4.00 vs 7.71ms，8×8 3.89 vs 7.69ms），快 ~1.5-2x。`tl-tuned`（按设备名选）与 `tl-rtx3060` 同源，结果一致。
- **大 T prefill（T>64）仍落后 faster3a**：T=128 `tl-rtx3060` 17.5ms vs faster3a 7.1ms。T>64 走 eager（graph 内存随 T 增长），fp16 GEMM 路径与 faster3a 的优化 kernel 有差距——这是所有 tilelang 路径的共性瓶颈。
- **对比 pure-torch**：T=1 快 7.3x（2.16 vs 15.78ms），T=64 快 57x（4.00 vs 228.87ms）。
- **`tl-fp16`（eager 无 Graph）是最慢项目路径**：小 T 时 launch 间隙主导（T=1 10.08ms），CUDA Graph 是全部收益来源。
- 编译 prefill 的结论：torch.compile 后 0.1B prefill 快 1.11-1.43x（T=8~256），但每个不同 T 都会重编译一张图（T=256 约 12 分钟，GPU 空闲），收益不抵成本，故 `prefill` 保持 eager。详见 docs/benchmarks/rtx3060.md。

### CUDA (MX450, sm_75)

- T=1 decode 时，graph_decoder 最快，说明 CUDA Graph 对单 token 解码的 launch 开销消除是有效的。
- **单 kernel prefill（fused_dplr_T）**：state 串行递推在 kernel 内、一次 launch 交付整个序列 + fp32io16 state。fp16 迁移前 prefill 曾**反超 faster3a_2607**（0.1B 1×128 37.6ms vs 87.8ms，快 2.34x）；fp16 迁移后因 Turing fp16 GEMM 病态慢退为与 faster3a 接近（1×128 92.0 vs 87.9ms）。**改用 checkpoint 原始 bf16 的 tl-bf16 后 prefill 再次全面反超 faster3a**：T=8 20.5 vs 24.2ms，T=128 39.8 vs 87.7ms（快 2.2x），T=32..128 均最快。
- 对比 pure_torch：0.1B 1×128 tl-fp16 快 14x（92.2 vs 1332.2ms），tl-bf16 快 33x（39.8 vs 1332.2ms）。
- T=1 decode 仍慢于 faster3a_2607（0.1B tl-fp16 15.4 vs 9.9ms，tl-bf16 18.3ms），因 fused kernel 逐 token dispatch 开销；graph_decoder（CUDA Graph）可弥补此差距。
- faster3a_2607 在 0.4B 上 T=64（150ms）反比 T=32（166ms）快，因其 chunk kernel 对不同序列长度有不同性能特征。
- **MX450 2GB 显存限制**：0.4B 正确性门控会同时加载 pure_torch 参考模型（合计 ~2.4GB > 2GB），导致 rwkv_tl 延迟虚高 10x（540ms vs 实际 56ms）。0.4B 数据不开 `--correctness-check` 采集。

### CPU

- CPU 上 rwkv_tl 与 pure_torch 接近，说明当前的 fused path 在 CPU 端没有明显优势。
- 在较大 prefill 场景下，rwkv_tl 的批处理收益仍然有限，结果与 CUDA 上的差异一致。

## 复现命令

```bash
# CUDA (0.1B)
RWKV_CHECKPOINT_PATH=...0.1b.pth RWKV_FAST_SCRIPT_PATH=.../faster3a_2607 .venv/bin/python script/benchmark_rwkv7.py --targets faster3a_2607,tl-fp16,pure-torch,graph_decoder --device cuda --cases 1x1,1x32,1x64,1x128 --warmup 5 --iters 10

# CPU (0.1B) - Albatross/graph_decoder 自动跳过
RWKV_CHECKPOINT_PATH=...0.1b.pth .venv/bin/python script/benchmark_rwkv7.py --targets faster3a_2607,tl-fp16,pure-torch --device cpu --cases 1x1,1x8,1x32 --warmup 1 --iters 3

# bf16 对比（checkpoint 原始 dtype）
RWKV_CHECKPOINT_PATH=...0.1b.pth .venv/bin/python script/benchmark_rwkv7.py --targets tl-fp16,tl-bf16,pure-torch --device cuda --cases 1x1,1x32,1x64 --warmup 5 --iters 10
```

更细的实验发现请见 [docs/benchmarks/rtx3060.md](../docs/benchmarks/rtx3060.md)。
