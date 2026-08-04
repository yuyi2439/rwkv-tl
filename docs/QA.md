# 常见问题 (QA)

## Q: torch.compile 时打印 `Not enough SMs to use max_autotune_gemm mode` 警告？

**A: 无害，不用管。** 当 `is_torch_compile=True`（默认）时 decode 会走 `torch.compile`，inductor 在
codegen 阶段会探测 GPU 是否适合 GEMM 模板/autotune（`is_big_gpu`），在 SM 数 < 68 的小 GPU
（如 MX450，20 SM）上打印这条 `log.warning`。autotune 本身默认就是关闭的，这个探测只是判断
"能不能用模板"，结果是小 GPU 跳过模板、走默认 GEMM 路径。不影响正确性，也不影响性能
（小 GPU 本来就不该用模板）。该警告每次进程首次编译时打印一次。

## Q: 测试时大量 `tilelang .../builder.py: DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type'...`

**A: tilelang 上游与 Python 3.13 的兼容问题，无害。** tilelang 的 eager builder 调用
`typing._eval_type` 时未传 `type_params`（Python 3.13 起弃用、3.15 移除），该警告来自
`site-packages/tilelang/.../builder.py`，非本项目代码。升级 tilelang 到新版本即可消除；不升级也不影响运行。

## Q: generate 输出一直重复同一句话（如"我是一个人"循环）？

**A: 这是贪心解码的退化循环，不是 bug。** 0.1B 小模型 + 无 repetition penalty 的贪心 argmax 容易陷入重复。
用 `generate(..., temperature=0.8, top_p=0.9, repetition_penalty=1.2)` 采样可改善
（`script/rwkv_chat.py` 已默认启用这些参数）。

## Q: `generate(stop=[tokenizer.encode("\n\nUser:")])` 偶尔不生效，回复里泄漏出 "User:"？

**A: 已知限制——token 精确匹配对子串 stop 不可靠。** `stop` 是按 **token id 序列精确相等** 判断的，
而模型生成 `"\n\nUser:"` 时的切分不一定等于 `tokenizer.encode("\n\nUser:")`。
实测模型会输出 `[..., 28329("…。\n"), 11("\n"), 24281("User"), 59(":")]`，而 greedy 编码是
`[261, 24281, 59]`（`261="\n\n"`），尾部永远不等于 stop 序列，于是泄漏出下一轮 "User: ..."。

不是 bug，是子串文本 stop 的固有局限。等 RWKV checkpoint 提供**专用对话停止 token** 后用那个
token id 做 stop（token 精确匹配就是对的）；不要依赖子串文本 stop。

## Q: bf16 权重转 fp16 会不会丢精度？

**A: 大数无损、极小的数会丢（下溢/精度退化），实测 0.05% 且全部无害。**

bf16 是 8 位指数 + 7 位尾数，fp16 是 5 位指数 + 10 位尾数。fp16 尾数更多所以**精度更高**，
但**指数范围小得多**：fp16 最小正正规数约 6e-5、最小次正规数约 6e-8，而 bf16 能表示到 1e-38。

对两个 checkpoint 的原始 bf16 权重做 `bf16→fp16→bf16` 往返测试：

| 模型 | 总 bf16 值 | 往返丢失 | 溢出(>65504→inf) | 丢失值幅值范围 |
|---|---|---|---|---|
| 0.1B | 191,084,544 | 84,287 (0.044%) | **0** | [7.7e-29, 7.60e-6] |
| 0.4B | 450,834,432 | 229,312 (0.051%) | **0** | [7.7e-29, 7.60e-6] |

结论：

- **无溢出**：没有任何 bf16 值超过 fp16 上限 65504。
- **丢失全部集中在 `|val| < 2^-17 ≈ 7.6e-6`**（fp16 次正规区间）。fp16 的 10 位尾数能精确覆盖
  bf16 的 7 位尾数，只要指数在 fp16 范围内就**逐位精确**（验证：`> 2^-17` 的丢失值个数 = 0）。
- **`< 6e-8` 的值下溢成 0**（fp16 最小次正规）：全模型仅约 5,507 个（0.0003%）。
- 大权重矩阵（receptance/key/output 等）往返误差 ~3e-8、相对误差 ~1e-7，可忽略。

极小值的分布（0.1B，1.91 亿非零值中 123,523 个 `< 2^-17`）：

- **总体分散**：163/402 个 tensor 含极小值，但多数只有 1~3 个；成规模的仅 `emb.weight`、
  `head.weight`、各层 `ffn.key/value.weight`（占比 0.05~0.08%），且张量内部分散在大量行列，
  每列占比 < 1%，无块状聚集。
- **唯一的集中**：`emb.weight` 最后 6 行（词表 65530~65535，padding/特殊 token）整行范数 ~2e-21，
  本质是零向量，fp16 下变精确 0，无害。
- 注意 `blocks.0.att.v1` 整个就是全 0 张量（不是极小值），也无需担心。

实际影响：一致性测试验证 rwkv_tl（fp16）与 pure_torch（bf16 原始）max_abs ~0.031、argmax 一致，
转换是安全的。复现：

```python
rt = t.half().to(torch.bfloat16)
lossy = (rt != t).sum().item()   # t 为 checkpoint 的 bf16 张量
```
