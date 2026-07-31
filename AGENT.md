# AGENT.md

Project guide for AI agents working on `rwkv-tl`.

## 项目概述

`rwkv-tl` 是 RWKV7 模型的 TileLang 高性能推理实现，目标是用 tilelang 融合 kernel + CUDA Graph 达到 Albatross（`rwkv7_fast_v3a.py`）80% 以上的性能。

- 模型：RWKV7-g1d（0.1B: C=768,H=12 / 0.4B: C=1024,H=16，N=64 固定）
- 精度：bfloat16（计算）+ float32（DPLR 累加）
- 路径：decode（T=1，CUDA Graph）+ prefill（T>1，GEMM 批处理）
- 依赖：tilelang>=0.1.12, torch, pytest

## 目录结构

```
src/rwkv_tl/
  __init__.py        # RWKV7 主类：make_TMIX/make_CMIX（decode）+ make_TMIX_batch/make_CMIX_batch（prefill）
  graph_decode.py    # GraphDecoder：CUDA Graph 捕获的逐 token 解码
  tokenizer.py       # BPE tokenizer
  kernels/
    _common.py       # 共享常量：HEAD_DIM=64, BLOCK=256, WARP=32, SERIAL=2, _SQRT_E
    lerp.py          # fused_lerp1/lerp6 + *_copy 变体（token-shift）
    gates.py         # fused_w_gate, fused_v_gate, fused_a_kk_k, fused_neg_kk_a
    dplr.py          # fused_dplr, fused_l2norm_neg_kk_a, fused_gn_rkrk
    gemm.py          # fused_rkv_gemm（prefill 批处理 GEMM，多路径 dispatch）
    __init__.py      # 统一导出
test/                # pytest：test_forward（端到端）, test_graph（GraphDecoder）, test_kernels
script/              # profile_prefill.py, benchmark_rwkv7.py, rwkv_chat.py
asset/               # rwkv_vocab_v20230424.txt
```

## 关键约束（必须遵守）

1. **不写兼容补丁**：不为旧代码加 compatibility shim，直接改。
2. **不臆造 API**：tilelang/torch API 必须查证后再用，不能猜。
3. **优先用现有实现**：项目内或标准库已有的函数优先于新写。
4. **Git 操作需显式许可**：commit/push 必须用户明确要求。
5. **不创建多余文件**：优先编辑现有文件，不主动建 .md/README。
6. **src/rwkv_tl 目录 docstring 用简洁英文**，无 'Callers' 段落；其他文档/交流用中文。
7. **不吞异常**：try 只用于真正可恢复的场景，不能掩盖真实问题。

## 性能优化策略

### 已实现

- **算子融合**（tilelang）：LERP6、w/v/a 门控、L2norm+neg_kk_a、GroupNorm+rkrk、DPLR 状态更新全部融合为单 kernel，fp32 累加后输出 bf16。
- **DPLR 融合**：`fused_dplr` 单 kernel 完成 `S = S*W + S@A@Bᵀ + V⊗K; y = S@R`，state 原地更新消除 copy_。
- **GEMV→GEMM 批处理**（prefill）：`make_TMIX_batch` 将 decode 的逐 token GEMV 改为 `[T,C] @ [C,C]` GEMM，T 维并行。
- **fused_rkv_gemm**：r/k/v 三个 GEMM 融合为一次 batched launch（见下）。
- **CUDA Graph**：`GraphDecoder` 捕获 decode 路径，消除 launch 开销，state 用固定地址 buffer。

### fused_rkv_gemm 多路径 dispatch（kernels/gemm.py）

| 路径 | 条件 | 实现 |
|---|---|---|
| tilelang T.gemm | CUDA sm_80+ | 单 kernel，blockIdx.z 批处理 3 个 matmul，T.gemm 走 TensorCore，target 编译为设备原生 arch |
| cuBLAS bmm | CUDA sm_75 / tilelang 编译失败 | torch.bmm 走 strided batched GEMM，fp32 累加，bit-exact |
| eager | CPU | stacked matmul |

- 权重在 `make_TMIX_batch` 闭包外预 stack 为 `[3,C,C]`，运行时零 stack 开销。
- `T.dynamic("C")` + `T.dynamic("T_LEN")` 动态形状，一套编译服务所有模型。
- sm_75 早退：`_gpu_supports_tl_bf16_gemm()` 检测 arch，避免无谓编译。

### 为什么 sm_75 上 tilelang T.gemm 不可用

sm_75 (Turing) 的 bf16 MMA 只支持 m16n8k8 + TransB=false，而 tilelang 推断的布局是 TransB=true，编译失败（官方 gemm_relu 示例也失败）。sm_80+ 支持 m16n8k16 + TransB，tilelang 路径自动启用。

## 测试与验证

### 数值一致性

- `test/test_forward.py`：端到端 logits 对比（0.1B + 0.4B）。
- `test/test_graph.py`：GraphDecoder vs 基准前向。
- `test/test_kernels.py`：各 fused kernel 单元测试。
- prefill 一致性：`forward_prefill` vs `forward`（sequential），独立 state，要求 bit-exact。

**重要**：测试时 `forward` 和 `forward_prefill` 必须用独立 state（`zero_state()` 各调一次），否则 state 污染会导致假阳性 diff。

### 性能基准

- `script/profile_prefill.py`：torch.profiler 分析 prefill 各阶段 GPU 时间分布。
- `script/benchmark_rwkv7.py`：对比 GraphDecoder vs Albatross。
- 环境变量 `RWKV_CHECKPOINT_PATH` 指定模型路径。

### 运行

```bash
cd /home/yuyi2439/rwkv/rwkv-tl
.venv/bin/python -m pytest test/ -v
RWKV_CHECKPOINT_PATH=...pth .venv/bin/python script/profile_prefill.py
```

## 性能现状（0.1B, CUDA Graph, sm_75）

### prefill vs Albatross

| T | rwkv-tl (us/tok) | Albatross (us/tok) | ratio |
|---|---|---|---|
| 16 | 1587 | 3841 | 242% |
| 32 | 1039 | 2673 | 257% |
| 64 | 660 | 1382 | 209% |
| 128 | 535 | 1340 | 250% |

rwkv-tl 超过 Albatross 的原因：GEMM 批处理 T 维（cuBLAS for SM120）+ Albatross 的 `wkv_seq` 是 kernel 内 T 串行 + Albatross 编译目标 sm_75。**超过 Albatross 不是 bug**，是路径差异。

### prefill 时间分布（T=32, 0.1B）

| 阶段 | 占比 | 说明 |
|---|---|---|
| GEMM/GEMV | 82.4% | 主瓶颈（aten::mm 71%, addmv 6.6%） |
| DPLR | 6.3% | 384 次（12层×32token），串行 |
| other | 10.0% | elementwise + memcpy |
| norm/gates | 1.4% | 已融合 |

## 已知限制

- **0.4B 精度**：`forward_prefill` 与 `forward` bit-exact（独立 state 下），但与 Albatross 的 argmax 不同（Albatross 用 fp16 + 不同 kernel 路径，非 rwkv-tl 问题）。
- **DPLR 串行**：prefill 的 DPLR 仍逐 token 循环（占 6.3%），未实现 chunk 并行。
- **chunk 并行未实现**：FlashRWKV/FLA 的 chunk 策略需要 WY 表示 + 块内矩阵求逆 + 6-kernel 编排，复杂度高。若实现，intra-chunk 必须用 `T.gemm`（TensorCore），不能像 FlashRWKV 那样标量 fmaf（否则在 SM120 上 chunk 反而更慢）。

## 开发规范

- 代码风格：Google-style docstring（src/rwkv_tl 用简洁英文），类型注解齐全。
- kernel 拆分：按功能分文件（lerp/gates/dplr/gemm），`__init__.py` 统一导出。
- 动态形状：所有 kernel 用 `T.dynamic` 参数化 C/H/T_LEN，不写死模型参数。
- CPU fallback：每个 kernel wrapper 都有 `if x.device.type != "cuda"` 分支，保证 CPU 可跑。
- 多 GPU arch：tilelang 路径用 try/except + arch 检测，sm_75 优雅降级到 cuBLAS。

## 参考资料

- Albatross 参考实现：`/home/yuyi2439/rwkv/Albatross/faster3a_2607/rwkv7_fast_v3a.py`
- FlashRWKV：`https://github.com/rwkv-rs/FlashRWKV`（chunk 在 SM120 上实测更慢，因 intra-chunk 用标量 fmaf 而非 MMA）
- FLA RWKV7：`https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/`（fused_recurrent + chunk 双路径，chunk 用 tl.dot TensorCore）
- tilelang gemm 示例：`.venv/.../tilelang/tools/pass_visualizer/examples/gemm_relu.py`
