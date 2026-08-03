# RTX 3060 验证测试记录

> 本文档记录当前版本在 RTX 3060 上的验证测试结果。使用中文撰写。
> 相关基准数据见 [benchmark_rwkv7.md](../script/benchmark_rwkv7.md)，实验发现见 [benchmark_rwkv7_experiments.md](benchmark_rwkv7_experiments.md)。

## 版本与环境

| 项目 | 值 |
|---|---|
| 代码版本 | commit `2f8c8f6`（stateless 重构后，`chore: bump version to 0.1.1`） |
| GPU | NVIDIA RTX 3060 (sm_86, 12GB GDDR6) |
| CUDA / PyTorch | CUDA 13.3 / PyTorch 2.13.0+cu130 |
| TileLang | 0.1.12 |
| 测试日期 | 2026-08-03 |

## 测试结果：13/13 通过

两个模型在 RTX 3060 上均通过完整测试套件（`pytest test/ -v`，13 项）。
该结果在 stateless 重构（`694962f`，含单 kernel `fused_dplr_T`、fp32 state、`is_torch_compile` 参数）后重新验证通过：

| 模型 | 参数 | 结构 | 结果 |
|---|---|---|---|
| rwkv7-g1d-0.1b-20260129-ctx8192 | C=768, H=12, L=12 | 13/13 PASSED | 通过 |
| rwkv7-g1d-0.4b-20260210-ctx8192 | C=1024, H=16, L=24 | 13/13 PASSED | 通过 |

测试项覆盖：
- `test_forward.py`：decode / prefill 与 pure_torch 参考的数值一致性（argmax、top-5、logit 差上限），decode 与 prefill 路径互相对齐。
- `test_graph.py`：GraphDecoder 与基线 forward 的一致性。
- `test_kernels.py`：fused_lerp6 与 6 次独立 LERP 的 bit-exact 对比。

运行命令：

```bash
RWKV_CHECKPOINT_PATH=~/rwkv/model/rwkv7-g1d-0.1b-20260129-ctx8192.pth .venv/bin/python -m pytest test/ -v
RWKV_CHECKPOINT_PATH=~/rwkv/model/rwkv7-g1d-0.4b-20260210-ctx8192.pth .venv/bin/python -m pytest test/ -v
```

## 未测试的模型

| 模型 | 说明 |
|---|---|
| rwkv7-g1i_preview3260-7.2b-20260716-ctx12288 | 权重 14.4GB bf16 > 12GB 显存，加载即 OOM；结构上（L=32, C=4096, H=64）与 g1d 代码兼容，但未运行测试 |
