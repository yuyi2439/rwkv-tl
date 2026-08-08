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

warmup=10, iters=20，正确性门控全过。`tl-fp16`/`tl-bf16`/`pure-torch` 均为 CUDA-Graph 包装
（decode + per-T prefill graph，`prefill_graph_max_t=1024`）。faster3a_2607 无 graph。
数据采集：2026-08-08，neo-kernel 分支（fused_rank_gemv/fused_gates 融合 + graph cap 1024）。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 5.94 | 168.26 |
| faster3a_2607 | 1×8 | 7.97 | 1003.96 |
| faster3a_2607 | 1×32 | 9.55 | 3352.16 |
| faster3a_2607 | 1×64 | 9.92 | 6450.33 |
| faster3a_2607 | 1×128 | 9.15 | 13988.56 |
| faster3a_2607 | 8×8 | 7.97 | 8026.84 |
| faster3a_2607 | 16×16 | 7.78 | 32899.07 |
| tl-fp16 | 1×1 | 2.36 | 424.13 |
| tl-fp16 | 1×8 | 2.92 | 2743.12 |
| tl-fp16 | 1×32 | 3.35 | 9562.51 |
| tl-fp16 | 1×64 | 3.81 | 16776.55 |
| tl-fp16 | 1×128 | 5.15 | 24868.20 |
| tl-fp16 | 8×8 | 3.88 | 16482.81 |
| tl-fp16 | 16×16 | 9.22 | 27770.64 |
| tl-bf16 | 1×1 | 2.08 | 480.12 |
| tl-bf16 | 1×8 | 3.24 | 2470.36 |
| tl-bf16 | 1×32 | 3.30 | 9703.46 |
| tl-bf16 | 1×64 | 4.07 | 15719.50 |
| tl-bf16 | 1×128 | 5.45 | 23503.28 |
| tl-bf16 | 8×8 | 4.10 | 15605.49 |
| tl-bf16 | 16×16 | 8.88 | 28813.67 |
| pure-torch | 1×1 | 3.74 | 267.08 |
| pure-torch | 1×8 | 5.93 | 1348.04 |
| pure-torch | 1×32 | 14.96 | 2139.03 |
| pure-torch | 1×64 | 26.81 | 2386.97 |
| pure-torch | 1×128 | 51.14 | 2502.95 |
| pure-torch | 8×8 | 26.67 | 2399.55 |
| pure-torch | 16×16 | 100.11 | 2557.10 |

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
> **sm_75 fp16 GEMM（m16n8k8）已实现**：`rwkv_tl.kernel.gemm` 为 sm_75 的 fp16 加了
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

warmup=10, iters=20，正确性门控全过（`tl-fp16` 对 pure-torch 门控）。`tl-fp16`/`pure-torch`
为 CUDA-Graph 包装（decode + per-T prefill graph，cap 1024）。faster3a_2607 无 graph。
数据采集：2026-08-08，neo-kernel 分支。

| 实现 | B×T | p50 (ms) | tok/s |
|---|---|---|---|
| faster3a_2607 | 1×1 | 6.57 | 152.18 |
| faster3a_2607 | 1×8 | 12.55 | 637.36 |
| faster3a_2607 | 1×32 | 15.74 | 2032.52 |
| faster3a_2607 | 1×64 | 15.69 | 4079.79 |
| faster3a_2607 | 1×128 | 13.58 | 9427.34 |
| faster3a_2607 | 16×16 | 16.48 | 15533.47 |
| tl-fp16 | 1×1 | 5.12 | 195.37 |
| tl-fp16 | 1×8 | 8.80 | 909.03 |
| tl-fp16 | 1×32 | 8.36 | 3826.46 |
| tl-fp16 | 1×64 | 10.55 | 6064.13 |
| tl-fp16 | 1×128 | 14.63 | 8751.44 |
| tl-fp16 | 16×16 | 26.25 | 9751.15 |
| pure-torch | 1×1 | 8.24 | 121.41 |
| pure-torch | 1×8 | 13.57 | 589.42 |
| pure-torch | 1×32 | 32.76 | 976.85 |
| pure-torch | 1×64 | 58.01 | 1103.23 |
| pure-torch | 1×128 | 107.86 | 1186.77 |
| pure-torch | 16×16 | 215.01 | 1190.63 |
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

- **tl-fp16 全面领先**（除 16x16 真 batch 外）：T=1..128 均快于 faster3a_2607
  （1×1 2.36 vs 5.94ms，1×8 2.92 vs 7.97ms，1×32 3.35 vs 9.55ms，1×64 3.81 vs 9.92ms，
  1×128 5.15 vs 9.15ms，8×8 3.88 vs 7.97ms），快 ~1.8-2.5x。
- **tl-bf16 已追平 tl-fp16**：旧记录 bf16 在 sm_86 慢 ~4x（1×1 10.58 vs 2.35ms），
  本次（neo-kernel 分支：fused gates + CUDA-Graph 全覆盖）bf16 与 fp16 持平
  （1×1 2.08 vs 2.36ms，1×128 5.45 vs 5.15ms）。"sm_86 上 fp16 tensor core 赢"的旧结论失效。
- **prefill graph cap 64→1024 是大 T 关键收益**：T=256（16×16）tl-fp16 从 17.15 →
  9.22ms，pure-torch T=128 从 506 → 51ms——这些 case 现在回放捕获图而非 ~2175 次 eager launch。
  残余大 T 成本是 GEMM 计算本身（见 TODO #3 chunk 并行 prefill）。
- **16×16 仍落后 faster3a（9.22 vs 7.78ms）**：faster3a 是真 batch `[16,16]` 并行，
  rwkv_tl 是 256 tokens 单序列串行递推；追平需真实 batch 支持。
- **0.4B（2026-08-08 重测）**：tl-fp16 在 T=1..64 领先 faster3a，1×128 已追平
  （14.63 vs 13.58ms）；旧记录 1×128 37.63ms 因 graph cap 64 未 graph 化，现 cap 1024
  生效后 14.63ms。16×16 仍落后（真 batch）。
- **对比 pure-torch**：T=1 快 1.6x（2.36 vs 3.74ms），T=64 快 7x（3.81 vs 26.81ms），
  T=128 快 10x（5.15 vs 51.14ms）。
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
