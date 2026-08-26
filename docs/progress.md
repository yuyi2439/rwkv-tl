# 进度与经验记录

从 TODO.md 迁出的非任务内容：落地进度、已关闭的死路。TODO.md 只保留任务。

## 落地进度

### 2026-08-25 · 0.2 API 重构 + Albatross 对齐

- 0.2 API 重构完成。
- w 衰减对齐 Albatross（delta = w-1 存储，S + S*delta 更新）。
- CUDAGraph decode zero-copy fast-path：capture 绑定调用方 State 地址 +
  snapshot/restore 基线，去掉每步 `token.item()` 同步，逐位验证等价。

### 2026-08-26 · W8A16 量化落地

- QTensor（W8A16：int8 权重 + fp16 per-group G 缩放、非对称）+ 离线量化器
  `script/quantize.py`，量化 checkpoint 由 RWKV7Weight 透明加载。
- head 投影换 W8A16 手写 GEMV：1.96x vs torch.mv，端到端 decode -15%。
- 权重-算子静态路由：`HeadOps/TmixOps/CmixOps`（kernel/ops.py），按权重类型
  （QTensor→w8a16 / tensor→fp16）构造时一次绑死，无运行时 isinstance；
  KernelOp 取代旧 BoundKernel。
- cmix decode 融合 W8A16：kWt/vWt int8 常驻显存（无持久 fp16 副本），
  prefill 里临时 dequant 用完即弃。
- 回归：0.1B top1 一致率 fp16/q8 = 100%，max|diff| 0.26（量化下限）。
  decode 6.50→5.46 ms/tok（-16%），显存峰值 381→286 MiB（-25%）。

## 已关闭的死路（勿重开，详见 .agent/performance.md）

- tilelang 内重写单行 GEMV 结构（row1_exact4 / [M,K] 布局 / THREADS 扫描
  / half2 双输出）：THREADS 32→256 完全平坦（74.6-77.1us），瓶颈是
  occupancy（72 blocks × 32 thr ≈ 14%），split-K 表达不了。
- int8 直接套在现有 [K,M] lane-per-output GEMV 上：probe 实测只 +8%
  （68.5 vs 74.6us，字节减半时间不变 = latency-bound），2x 带宽收益
  无法兑现，需先解决并行度。
- DPLR 状态更新优化（DeltaLog 等）：实测仅占单层 4.5%、全 decode 1.4%，
  且 fp32 state 下无精度动因。
- 头投影 fp16 重写：torch.mv (CUTLASS) 已最优，tilelang 重写慢 4.4x。
