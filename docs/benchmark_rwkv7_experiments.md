# RWKV7 benchmark experiments

这份文件收录比主报告更细的实验发现、分析和后续建议，不用于存放纯粹的复现命令或原始结果表格。

## torch.compile 相关实验

### 结论

- 在 MX450 上，torch.compile 并没有带来收益，甚至明显变慢。
- 主要原因是该硬件的 sm_75 架构不适合 bf16 编译路径，inductor 会退回到更慢的 fallback 路径。
- 该结果不应被解释为 RTX 30 系列或更高架构上的通用结论。

### 观察

- `forward` / `forward_prefill` 的图断裂数在实验中为 0，说明 custom op 包装可以避免编译时的图断裂问题。
- decode 路径中，`Tensor.item()` 仍会引入一次同步，因而无法完全消除所有图断裂。
- 数值一致性保持为 bit-exact。

### 进一步建议

- 对 decode-only 的 compile 策略可以保留，但不建议把它作为默认的性能优化手段。
- 对 prefill 路径，当前更值得优先做的是 kernel 级别和访存路径的优化，而不是继续依赖 torch.compile。

## 运行与环境注意事项

- 这类大规模 benchmark 在显存和内存受限的机器上容易触发 OOM，建议按 case 分开运行。
- 对于长时间扫描，优先使用独立进程和日志文件方式执行，避免单次前台进程被中断。
- 未来在 RTX 3060 上验证时，需要把 MX450 的结论视为相对参考，而不是最终结论。
