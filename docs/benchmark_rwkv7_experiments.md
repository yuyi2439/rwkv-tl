# RWKV7 benchmark experiments

这份文件收录比主报告更细的实验发现、分析和后续建议，不用于存放纯粹的复现命令或原始结果表格。

## torch.compile 相关实验

### 结论

- 在 MX450 上，torch.compile 并没有带来收益，甚至明显变慢。
- 主要原因是该硬件的 sm_75 架构不适合 bf16 编译路径，inductor 会退回到更慢的 fallback 路径。
- 该结果不应被解释为 RTX 30 系列或更高架构上的通用结论。

### RTX 3060 (sm_86) 验证

目标卡上重新验证了编译路径，结论：

- **decode (`decode`)**: graph breaks = 0（custom op 包装生效）。但 compiled 与 eager 几乎相同甚至更慢（0.1B: 44-49 vs 43-46 ms/token，speedup ~0.88-1.04x），没有额外收益。
- **prefill (`prefill`)**: 先把 `make_TMIX_batch` 的 `fused_rkv_gemm` / `fused_dplr` 接线到 custom op 后，graph breaks 从 3 降到 0。编译后 0.1B 各 T 均更快：

  | T | eager ms | compiled ms | speedup | 首次编译耗时 |
  |---|---:|---:|---:|---:|
  | 8 | 43.7 | 30.6 | 1.43x | 22.6 s |
  | 16 | 70.4 | 50.8 | 1.39x | 31.4 s |
  | 32 | 128.2 | 102.7 | 1.25x | 60.7 s |
  | 64 | 227.5 | 186.5 | 1.22x | 126.9 s |
  | 128 | 479.1 | 398.2 | 1.20x | 289.5 s |
  | 256 | 841.6 | 759.8 | 1.11x | 702.1 s |

- **决策**: prefill 保持 eager，不启用 `maybe_torch_compile`。原因：DPLR 串行循环按 T unroll，每个不同 prompt 长度都重编译一张新图，编译耗时随 T 快速增长（T=256 约 12 分钟，期间 GPU 空闲），而稳态收益仅 1.11-1.43x。对长上下文 / 动态长度不划算。`make_TMIX_batch` 也回退到直接调 raw kernel（不经过 custom op，避免 dispatch 开销）。
- `_compat.maybe_torch_compile` 的缓存键改为 `f"_{fn.__name__}_impl"`，支持同一实例上多个方法各自编译（本次仅 `decode` 使用）。

### 观察

- `forward` / `prefill` 的图断裂数在实验中为 0，说明 custom op 包装可以避免编译时的图断裂问题。
- decode 路径中，`Tensor.item()` 仍会引入一次同步，因而无法完全消除所有图断裂（`forward` 单 token 分支在 __init__.py:292 处 graph break，graphs=2/breaks=1，但 `forward` 不编译所以无实际影响）。
- 数值一致性：compiled prefill 与 eager 的 max_diff=0.19，argmax 一致。
- 手动 `torch._dynamo.explain(m.forward)` 时，若 `_decode_impl` 尚未预热，`maybe_torch_compile` 包装器会在 dynamo 追踪内递归 `torch.compile`（RecursionError）。运行时首次真实调用会缓存 impl，正常流程不受影响。

### 进一步建议

- 对 decode-only 的 compile 策略可以保留（graph 干净、无额外编译成本），但不把它当作默认性能手段。
- 对 prefill 路径，当前更值得优先做的是 kernel 级别和访存路径的优化，而不是继续依赖 torch.compile。
- 若未来启用编译 prefill，考虑按 (T) 缓存编译图或限定固定长度，避免扫表场景的重复编译。

## benchmark_rwkv7.py 扫描注意事项

- `prefill` 一旦被 `maybe_torch_compile` 包装，benchmark 每个不同 T 的 case 都会触发一次全新编译（0.4B 的 16×16 要 20+ 分钟），GPU 空闲、看起来像死锁。benchmark 现在通过 `_eager_dispatch` 走 eager 路径测实现本身；编译收益单独用脚本测。
- 大规模 benchmark 在显存和内存受限的机器上容易触发 OOM，建议按 case 分开运行。
- 对于长时间扫描，优先使用独立进程和日志文件方式执行，避免单次前台进程被中断。

## decode 全融合单 kernel 的可行性评估（RTX 3060, 0.1B）

背景：曾想用 tilelang 把 decode 的逐层 Python 循环（`for TM, CM in self.layers`）融合成一个动态层数的单 kernel，消除 ~130 次 kernel launch。

测量结论：
- eager decode 13.1 ms/token（wall），纯 GPU 11.1 ms；单次 decode 有 27 次 launch（~130 个小 kernel），主要为 GEMV（~1 ms）、`_fused_rkv_gemm`（12 次）、`_impl` reduce（95 次）、layernorm、elementwise。
- **GraphDecoder（CUDA Graph）已到 1.63 ms/token**，且 profiler 显示 GPU kernel 总执行仅 1.55 ms、launch 间隙仅 ~0.08 ms。即 launch 开销已被 CUDA Graph 几乎完全消除。
- 因此"全融合单 kernel"的理论上限 ≈ 1.55 ms，相对 GraphDecoder 只省 ~0.08 ms，收益极小。真正剩余成本是 ~1.5 ms 的 GEMV 计算本身，而非 launch。

技术验证（tilelang）：
- `T.gemm` 要求 M ≥ 16（MMA tile），decode 是 M=1 的 GEMV，不能直接用；手写 block-per-row + warp_reduce 的 GEMV 正确且比 cuBLAS `mv` 快（11.3 vs 17.4 μs @ C=768）。
- `sync_grid` 需要 cooperative launch：grid=768 blocks × 32 threads 在 RTX 3060 上报 `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`，动态层数大网格不可行。
- `@T.macro` 能复用 tilelang 代码片段（编译期内联，多 kernel 共享），但宏内**不能用 Python 条件分支切换张量索引**（`if batched:` 会被 tilelang 当运行时条件，生成错误代码）。因此 DPLR 的两个 kernel（T=1 输入 `[H,N]`、T>1 输入 `[T,H,N]`）索引布局不同，无法共用同一个宏；统一成 3D 布局需改 `fused_dplr` 公开契约，不值得。结论：DPLR 保持两处独立实现，不抽宏。

方向决策：
- **rwkv-tl 长期目标支持训练**。CUDA Graph 本质 inference-only：捕获前向启动序列、不重建 autograd 图（replay 不记录梯度，固定 buffer 与 autograd 动态建图冲突）。全融合单 kernel 并非本质 inference-only（任何自定义 kernel 都要显式 backward），但整层融合会让训练困难：需手写整层 backward（含串行 DPLR 递推的时间反向）并手工保存各算子的中间张量，远超 per-op custom op 路径的工作量。
- 正确的训练路径是保留 `torch.library.custom_op`（`torch.ops.rwkv_tl.*`），通过 `register_autograd` 定义 backward。
- 结论：decode 的优化应聚焦 GraphDecoder 那 1.5 ms GEMV 计算本身（如 output/FFN/低秩 gates 的 GEMV 优化），而不是全融合层数。不要为"融合"投入，除非能保持 autograd 能力。
