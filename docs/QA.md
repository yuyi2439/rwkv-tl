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
