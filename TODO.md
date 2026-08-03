# TODO

优先级从高到低。完成一项就删掉对应条目。

## 3. 避免 bf16 GEMM（如果可行）

- **背景**：bf16 MMA 仅 sm_80+ 支持（sm75 回退 cuBLAS bmm），且 bf16 尾数 7-bit 精度低于
  fp16 的 10-bit。Albatross 全用 fp16（+fp32 累加），兼容 sm75 且精度更高。
- **做法**：评估把 r/k/v 投影 GEMM 改走 fp16（权重转 fp16 + fp32 累加），或至少在 sm80+ 用
  fp16 T.gemm。若收益不明确则记录结论后关闭。

## 4. kernel 迁移到 @tilelang.jit

- **目标**：把 dplr/gates/lerp/gemm 里的 `@T.prim_func` + `tilelang.compile` 改成
  `@tilelang.jit`（lazy 模式，函数内 `@T.prim_func` 并 `return main`）。
- **注意**：lazy 模式支持 in-place（已验证 dplr 原地更新正确）；eager 模式对 in-place 损坏，不要用。
- **收益**：懒编译（import 不再编译 CUDA kernel）、代码更简洁。
