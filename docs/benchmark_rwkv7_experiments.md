# RWKV7 benchmark experiments

这份文件收录比主报告更细的实验发现、分析和后续建议，不用于存放纯粹的复现命令或原始结果表格。

## torch.compile 相关实验

### 结论

- 在 MX450 上，torch.compile 并没有带来收益，甚至明显变慢。
- 主要原因是该硬件的 sm_75 架构不适合 bf16 编译路径，inductor 会退回到更慢的 fallback 路径。
- 该结果不应被解释为 RTX 30 系列或更高架构上的通用结论。

### RTX 3060 (sm_86) 验证

目标卡上重新验证了编译路径，结论：

- **decode (`run_one`)**: graph breaks = 0（custom op 包装生效）。但 compiled 与 eager 几乎相同甚至更慢（0.1B: 44-49 vs 43-46 ms/token，speedup ~0.88-1.04x），没有额外收益。
- **prefill (`forward_prefill`)**: 先把 `make_TMIX_batch` 的 `fused_rkv_gemm` / `fused_dplr` 接线到 custom op 后，graph breaks 从 3 降到 0。编译后 0.1B 各 T 均更快：

  | T | eager ms | compiled ms | speedup | 首次编译耗时 |
  |---|---:|---:|---:|---:|
  | 8 | 43.7 | 30.6 | 1.43x | 22.6 s |
  | 16 | 70.4 | 50.8 | 1.39x | 31.4 s |
  | 32 | 128.2 | 102.7 | 1.25x | 60.7 s |
  | 64 | 227.5 | 186.5 | 1.22x | 126.9 s |
  | 128 | 479.1 | 398.2 | 1.20x | 289.5 s |
  | 256 | 841.6 | 759.8 | 1.11x | 702.1 s |

- **决策**: prefill 保持 eager，不启用 `maybe_torch_compile`。原因：DPLR 串行循环按 T unroll，每个不同 prompt 长度都重编译一张新图，编译耗时随 T 快速增长（T=256 约 12 分钟，期间 GPU 空闲），而稳态收益仅 1.11-1.43x。对长上下文 / 动态长度不划算。`make_TMIX_batch` 也回退到直接调 raw kernel（不经过 custom op，避免 dispatch 开销）。
- `_compat.maybe_torch_compile` 的缓存键改为 `f"_{fn.__name__}_impl"`，支持同一实例上多个方法各自编译（本次仅 `run_one` 使用）。

### 观察

- `forward` / `forward_prefill` 的图断裂数在实验中为 0，说明 custom op 包装可以避免编译时的图断裂问题。
- decode 路径中，`Tensor.item()` 仍会引入一次同步，因而无法完全消除所有图断裂（`forward` 单 token 分支在 __init__.py:292 处 graph break，graphs=2/breaks=1，但 `forward` 不编译所以无实际影响）。
- 数值一致性：compiled prefill 与 eager 的 max_diff=0.19，argmax 一致。
- 手动 `torch._dynamo.explain(m.forward)` 时，若 `_run_one_impl` 尚未预热，`maybe_torch_compile` 包装器会在 dynamo 追踪内递归 `torch.compile`（RecursionError）。运行时首次真实调用会缓存 impl，正常流程不受影响。

### 进一步建议

- 对 decode-only 的 compile 策略可以保留（graph 干净、无额外编译成本），但不把它当作默认性能手段。
- 对 prefill 路径，当前更值得优先做的是 kernel 级别和访存路径的优化，而不是继续依赖 torch.compile。
- 若未来启用编译 prefill，考虑按 (T) 缓存编译图或限定固定长度，避免扫表场景的重复编译。

## benchmark_rwkv7.py 扫描注意事项

- `forward_prefill` 一旦被 `maybe_torch_compile` 包装，benchmark 每个不同 T 的 case 都会触发一次全新编译（0.4B 的 16×16 要 20+ 分钟），GPU 空闲、看起来像死锁。benchmark 现在通过 `_eager_dispatch` 走 eager 路径测实现本身；编译收益单独用脚本测。
- 大规模 benchmark 在显存和内存受限的机器上容易触发 OOM，建议按 case 分开运行。
- 对于长时间扫描，优先使用独立进程和日志文件方式执行，避免单次前台进程被中断。
