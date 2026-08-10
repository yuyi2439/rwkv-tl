# RTX 3060 验证测试记录

> 本文档记录当前版本在 RTX 3060 上的验证测试结果。使用中文撰写。
> MX450 (sm_75) 的验证记录见 [validation_mx450.md](validation_mx450.md)。

## 三模型 fp16 vs faster3a 基准（2026-08-10，a8e2ef7）

### 版本与环境

| 项目 | 值 |
|---|---|
| 代码版本 | 分支 `neo-kernel`，commit `a8e2ef7`（use neo kernels） |
| GPU | NVIDIA GeForce RTX 3060 (sm_86, 12GB) |
| TileLang | 0.1.13 |
| 测试日期 | 2026-08-10 |

三个模型（0.1B g1d / 0.4B g1d / 1.5B g1i）分别跑 `tl-fp16`（本项目，graph 包装）
与 `faster3a_2607`（Albatross，无 graph），独立进程串行，warmup=10, iters=30。
结果见 [benchmarks/rtx3060.md](benchmarks/rtx3060.md)「neo kernels 基线」章节。

要点：
- prefill T=8/32/64 三模型均快 faster3a 1.6-2.4x。
- decode 1x1 回归：0.1B tl-fp16 graph 4.15ms > eager 3.44ms，1.5B 18.93ms vs faster3a
  7.91ms（慢 2.4x）。单 kernel 不慢（tmix_decode 0.13ms / cmix_decode 0.12ms @ 0.1B），
  疑为 CUDAGraph decode 的 State copy-in/out（36 次小 copy_）+ `.item()` 同步主导。
- 1.5B (g1i) 加载 + 正确性 + 完整 benchmark 均通过（TODO #4 完成）。

### bf16 回归（a8e2ef7 新引入，未修）

`test_forward.py::test_bf16_consistent` **失败**：`tmix_decode` 的 bf16 编译报
`Cannot find var remap for xr`（`unsupported_dtype_legalize.cc:713`）。与 AGENT.md
记录的 MX450 sm_75 bf16 错误同型，但**这次发生在 RTX 3060 (sm_86)**——是 a8e2ef7
新 `tmix_decode` 的 bf16 路径 bug，非硬件限制（fp16 全套 25 passed）。fp16 不受影响。
排查方向：`tmix_decode`/`gemv_macro` 的 bf16 dtype remap。

## neo CMIX kernel + recompute 策略验证（2026-08-10）

### 版本与环境

| 项目 | 值 |
|---|---|
| 代码版本 | 分支 `neo-kernel`，commit `cc09d44`（neo cmix kernels 补全） |
| GPU | NVIDIA GeForce RTX 3060 (sm_86, 12GB) |
| TileLang | 0.1.13 |
| PyTorch | 2.13.0+cu130 |
| 测试日期 | 2026-08-10 |

### 完整测试套件

`RWKV_CHECKPOINT_PATH=...0.1b.pth .venv/bin/python -m pytest test/` → **24 passed, 1 skipped**。
（本轮在重建的 `.venv` 上执行：conda 路径 miniconda3 → `.miniconda3` 重命名导致链接断裂，
`uv sync` 后以 python3.14 重建，全套仍通过。）

`test/test_neo_cmix.py` 5 项在 3060 上全部通过（C=768 fp16，decode 与 prefill 相对
eager CMIX 链 max_abs < 0.002）。

### recompute 开/关对比（cmix_prologue_prefill）

`cmix_prefill_prologue_macro` 的 token-shift 源策略由 `recompute` 参数选择：
- **recompute=True（默认）**：2 kernel，block 内重算 `LN_pre(x0[n-1])`（只读不可变
  `x0`），LN+lerp 融进 1 kernel；代价是 LN 计算量翻倍、省 1 次 launch。
- **recompute=False**：3 kernel（全量 LN → lerp 读 `x_ln[n-1]` → 拷回 `prev_x`）。

3060 本机实测（C=768 fp16，整链 `cmix_prefill`，5 次独立 run 取中位数）：

| LEN | recompute=True | recompute=False | 加速比 | 胜者 |
|---|---:|---:|---:|---|
| 32 | 0.142 ms | 0.147 ms | 1.04x | True |
| 64 | 0.140 ms | 0.148 ms | 1.06x | True |
| 128 | 0.142 ms | 0.222 ms | 1.57x | True |
| 256 | 0.155 ms | 0.166 ms | 1.08x | True |
| 512 | 0.252 ms | 0.251 ms | 1.00x | 持平 |

- **recompute=True 在 sm_86 上全面胜出或持平**，与 MX450 (sm_75) 的 14-32% 结论一致；
  T=512 打平（大 T 时 GEMM 占主导，prologue 差异被摊薄）。默认值正确，无需按硬件分支。
- 正确性：两种实现整链输出 max_abs 0.002（fp16 ULP 级），`prev_x` 状态完全一致，
  数值等价。

运行命令：

```bash
.venv/bin/python -m pytest test/test_neo_cmix.py -v
RWKV_CHECKPOINT_PATH=...0.1b.pth .venv/bin/python -m pytest test/
# recompute 对比
.venv/bin/python /tmp/opencode/rwkv-tl-bench/bench_recompute2.py
```
