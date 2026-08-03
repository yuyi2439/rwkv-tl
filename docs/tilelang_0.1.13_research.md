# TileLang 0.1.13 版本调研

> 2026-08-03，基于 GitHub release notes（v0.1.12 → v0.1.13，138 commits，2026-07-08 → 2026-08-02，其中 79 个 bug 修复）与本项目在 MX450 (sm_75) / RTX 3060 (sm_86) 上的实测。

## 版本核心变更摘要

1. **多后端语言方言重构**（#2734）— 移除运行时激活的 language facade，改为静态 per-backend re-export（`from tilelang.cuda.language import *`）。
2. **SM120 (Blackwell) NVF4 block-scale MMA**（#2364）— `T.mma_gemm_blockscaled`，SM120 上 8192³ ≈ 1527 TFLOPS。
3. **Metal 4 (M5) cooperative-tensor GEMM**（#2252）。
4. **任意 TMEM layout**（#2785）。
5. **TIR 源码位置注入**（#2751）— 编译错误信息更友好。

## 与本项目相关的改动

### sm_75 GEMM FMA fallback（#2811）— 实测无收益，保持现有 dispatch

0.1.13 起 `T.gemm` 能在 sm_75 上编译（此前 bf16 TransB MMA 在 sm_75 上无法编译，回退 cuBLAS bmm）。
在 MX450 (sm_75, C=768, T=128) 上实测 prefill r/k/v GEMM：

| 路径 | 耗时 |
|---|---|
| cuBLAS bmm（当前 sm_75 路径） | 0.546 ms |
| 3× 独立 torch.mm | 0.576 ms |
| tilelang `T.gemm`（0.1.13 sm_75 fallback） | **0.614 ms** |

tilelang 的 sm_75 fallback 走 FMA（非 TensorCore），比 cuBLAS 慢。**结论：sm_75 继续走 cuBLAS bmm，`_gpu_supports_tl_bf16_gemm()` 的 sm_80+ 门控保持不变。**

### `T.__exp` 修正（#2696）— 不影响本项目

release notes 指出 `T.__exp` 此前计算 `2**x`（应为 `e**x`），0.1.13 修正。我们的 gate kernel 用 `T.exp`
（非 `T.__exp`），且在 0.1.12/0.1.13 上都与 pure_torch 的 `torch.exp` bit-exact（0.1B max_abs=0.0），
说明本就正确，无需处理。

### JIT executable 跨 launch 复用（#2686）— 间接相关

本项目当前用 eager `tilelang.compile` + `functools.cache`（每进程编译一次），不受影响。
该收益只在迁移到 `@tilelang.jit` lazy 模式后（见 TODO #4）才有意义。

### 其它不相关

- **warp_reduce int64 截断修复**（#2782）：DPLR 的 warp reduce 是 fp32，不涉及 int64。
- **Pre-SM80 bf16 `__hfma` fallback**（#2769）：本项目 elementwise kernel 已 fp32 计算，非收益点。
- **SM120 NVF4 / TMEM layout / Metal5 / IKET profiler / fp32x2 reducer**：与 sm_86 解码无关；
  fp32x2 reducer 对 3060 上大 fused decode kernel 的归约精度可能有参考价值（推测，未验证）。

## 结论

0.1.13 对本项目**没有带来明确优化**：唯一新增能力（sm_75 GEMM）实测反而更慢；`__exp` 修复未踩坑；
测试无回归（13/13，0.1B bit-exact，0.4B max_abs=0.5）。**无需改动代码**。

> 若后续在 RTX 3060（或其它显卡）上有必要复核以上任何一项，可在对应显卡上按本文档的实测方法重跑。
