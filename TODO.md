# TODO

优先级从高到低。完成一项就删掉对应条目。

## 3. 避免 bf16 GEMM（如果可行）

- **背景**：bf16 MMA 仅 sm_80+ 支持（sm75 回退 cuBLAS bmm），且 bf16 尾数 7-bit 精度低于
  fp16 的 10-bit。Albatross 全用 fp16（+fp32 累加），兼容 sm75 且精度更高。
- **做法**：评估把 r/k/v 投影 GEMM 改走 fp16（权重转 fp16 + fp32 累加），或至少在 sm80+ 用
  fp16 T.gemm。若收益不明确则记录结论后关闭。
