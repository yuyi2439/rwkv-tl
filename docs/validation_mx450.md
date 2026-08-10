# MX450 验证测试记录

> 本文档记录当前版本在 MX450 上的验证测试结果。使用中文撰写。

## neo CMIX kernel 完成度验证（2026-08-10）

### 版本与环境

| 项目 | 值 |
|---|---|
| 代码版本 | 分支 `neo-kernel`，commit `e13a95b` 基础上补全 `kernel/neo/cmix.py` |
| GPU | NVIDIA GeForce MX450 (sm_75, 2GB) |
| CUDA / PyTorch | CUDA 13.3 |
| TileLang | 0.1.13 |
| 测试日期 | 2026-08-10 |

### 本次补全的内容

`kernel/neo/cmix.py` 之前未完成的部分：

1. **decode 残差 epilogue 未接线**：`_add_residual` 引用了不存在的全局 `residual`；
   `gemv_macro` 的 epilogue 文档签名是 `(acc, index, residual)` 但只传 2 参。
   修复：`gemv_macro` 新增可选 `residual` 张量参数（`kernel/neo/gemv.py`），
   down GEMV 以 `residual=x0` 调用（`cmix_main_decode_macro`）。
2. **decode prologue 张量形状不匹配**：`cmix_prologue_macro` 声明 `x0: [LEN, C]`，
   但 `cmix_decode` 传 1D `[C]` 单 token。修复：按 `LEN == 1` 区分 1D/2D 分支。
3. **token-shift 语义回归**：重构把 shift 源从 `x_ln[n-1]`（前一 token 的 LN 输出，
   与 eager `make_CMIX_batch` 一致）改成了 `x0[n-1]`（原始 token），数值错误放大
   到 ~1.75。修复为读 `x_ln[n-1]`。
4. **跨 block 竞争**：原实现 LN 与 lerp 同 kernel，block n 读 `x_ln[n-1]`
   （由 block n-1 写入）。修复：multi 路径拆成 LN kernel → lerp kernel 两段，
   前段完成后才 launch 后段；decode 单 token 无竞争，仍单 kernel。

### 测试结果：5/5 新增测试通过，全套 24/24 通过

`test/test_neo_cmix.py` 新增 5 项（C=768 fp16）：

| 测试 | 形状 | 结果 |
|---|---|---|
| `test_cmix_decode[42]` | 单 token C=768 | max_abs ~0.0008 |
| `test_cmix_decode[43]` | 单 token C=768 | max_abs ~0.0009 |
| `test_cmix_prefill[42-64-32]` | LEN=64, block=32 | max_abs ~0.0015 |
| `test_cmix_prefill[43-128-32]` | LEN=128, block=32 | max_abs ~0.0016 |
| `test_cmix_prefill[42-256-64]` | LEN=256, block=64 | max_abs ~0.0020 |

参考为 eager CMIX 链（LN_pre → shift lerp → relu²(x@kWt)@vWt + x0），
shift 源与 `_rwkv7_base.make_CMIX_batch` 一致；`prev_x` 状态更新误差 ≤ 0.00003。

另验证 C=1024（0.4B 形状）：decode / multi（LEN=128）均通过，max_abs < 0.0024。
`gemv` 独立 kernel 回归通过。

完整套件（含 checkpoint 0.1B）：
`RWKV_CHECKPOINT_PATH=... .venv/bin/python -m pytest test/` → **24 passed, 1 skipped**。

运行命令：

```bash
.venv/bin/python -m pytest test/test_neo_cmix.py -v
RWKV_CHECKPOINT_PATH=~/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth .venv/bin/python -m pytest test/
```

注意：MX450 (sm_75) 无 bf16 tensor core，bf16 路径需在 sm_80+（RTX 3060）上另行验证。
