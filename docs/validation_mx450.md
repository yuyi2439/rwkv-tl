# MX450 验证测试记录

> 本文档记录当前版本在 MX450 上的验证测试结果。使用中文撰写。

## neo CMIX kernel 完成度验证（2026-08-10）

### 版本与环境

| 项目 | 值 |
|---|---|
| 代码版本 | 分支 `neo-kernel`，commit `e13a95b` 基础上补全 `kernel/cmix.py` |
| GPU | NVIDIA GeForce MX450 (sm_75, 2GB) |
| CUDA / PyTorch | CUDA 13.3 |
| TileLang | 0.1.13 |
| 测试日期 | 2026-08-10 |

### 本次补全的内容

`kernel/cmix.py` 之前未完成的部分：

1. **decode 残差 epilogue 未接线**：`_add_residual` 引用了不存在的全局 `residual`；
   `gemv_macro` 的 epilogue 文档签名是 `(acc, index, residual)` 但只传 2 参。
   修复：`gemv_macro` 新增可选 `residual` 张量参数（`kernel/gemv.py`），
   down GEMV 以 `residual=x0` 调用（`cmix_decode_main_macro`）。
2. **decode prologue 张量形状不匹配**：`cmix_prologue_macro` 声明 `x0: [LEN, C]`，
   但 `cmix_decode` 传 1D `[C]` 单 token。修复：按 `LEN == 1` 区分 1D/2D 分支。
3. **token-shift 语义回归**：重构把 shift 源从 `x_ln[n-1]`（前一 token 的 LN 输出，
   与 CMIX token-shift 语义一致）改成了 `x0[n-1]`（原始 token），数值错误放大
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
shift 源与 CMIX token-shift 语义一致；`prev_x` 状态更新误差 ≤ 0.00003。

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

## neo TMIX decode kernel（2026-08-10）

### 本次新增的内容

`kernel/tmix.py` 实现单 token TMIX 全链 `tmix_decode`（LN_pre → 6 路
token-shift → r/k/v 投影 → 4 路低秩 gate → L2-norm + DPLR 递归 → GroupNorm +
r·k·r_k 残差 → g 门控输出投影 + x0 残差），参考 `rwkv7_torch.time_mix`，
逻辑复制自 `kernel/{dplr,gates}.py`（不引用外部老 kernel）。

结构（prologue + main 两个 macro，共 11 个 kernel）：
- `tmix_decode_prologue_macro`：LN_pre + 6 shift + `prev_x` 原地更新（1 kernel）
- `tmix_decode_main_macro`：rkv 3 个 GEMV + packed rank GEMV + gate math +
  L2-norm + DPLR + GroupNorm + out GEMV

LN_pre 计算提取到 `kernel/ln.py`（`ln_pre_row_macro`），cmix/tmix 共用。

状态接口：`prev_x`（shift 源）与 `rnn`（DPLR，fp32）原地更新；`v_first`
（v 残差门状态）跨 token 传递，首个 token 传 `first=1` 跳过 v 门并初始化
`v_first`（与 `rwkv7_torch.time_mix` 语义一致）。

### 测试结果：2/2 新增测试通过，全套 26/26 通过

`test/test_neo_tmix.py`（真实 0.1B checkpoint，C=768 H=12 低秩门 rank
32/64/64/128，参考为 `rwkv7_torch.time_mix`）：

| 测试 | 结果 |
|---|---|
| `test_tmix_decode[42]` | 4 步链 out max_abs ≤ 0.02, rnn ≤ 0.014 |
| `test_tmix_decode[43]` | 4 步链 out max_abs ≤ 0.02, rnn ≤ 0.014 |

`prev_x` 状态逐位一致；`v_first` 跨 token 传递正确。

运行命令：

```bash
RWKV_CHECKPOINT_PATH=~/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth .venv/bin/python -m pytest test/test_neo_tmix.py -v
```

### 性能：tmix_decode vs rwkv7_torch.time_mix（MX450, 0.1B 单层）

单 token decode 单层 TMIX，CUDA events + median（1000 iters），3 次独立运行：

| 实现 | 耗时 | 
|---|---|
| `rwkv7_torch.time_mix`（torch 参考） | 0.42-0.49 ms |
| neo `tmix_decode`（融合） | 0.13-0.19 ms |
| 加速比 | **2.5-3.4x** |

收益来源：一次 host 调用顺序 launch 全部 kernel（省 Python dispatch 与中间
`[C]`/rank 张量的分配/释放），以及 prologue 把 LN+6-shift 融成 1 个 kernel。
MX450 热节流导致各次运行波动，但加速比稳定在 ~2.5x 以上。

## 与 Albatross (faster3a_2607) 对比（2026-08-10, MX450 0.1B, eager）

`script/benchmark_rwkv7.py`，eager（无 CUDA Graph），warmup=20 iters=100，
对比 `tl-fp16`（重构后 RWKV7TL）vs faster3a_2607：

| Case | faster3a_2607 | tl-fp16 | 加速比 |
|---|---|---|---|
| 1x1 decode | 15.28 ms | 9.12 ms | 1.68x |
| 1x8 | 24.63 ms | 19.65 ms | 1.25x |
| 1x32 | 43.88 ms | 23.34 ms | 1.88x |
| 1x128 | 87.80 ms | 56.69 ms | 1.55x |
| 1x256 | 169.48 ms | 120.66 ms | 1.40x |
| 1x512 | 258.93 ms | 236.69 ms | 1.09x |

结论：decode 与小-中 prefill 领先 1.25-1.88x（融合 kernel 一次 host 调用 +
省中间张量分配）；大 T 趋平（GEMM 计算主导）。均为 eager 数值，未叠 CUDA
Graph（Graph 会进一步拉开 decode 差距）。

### 0.4b（C=1024, H=16, L=24）对比（均含 CUDA Graph）

`tl-fp16` 与 faster3a_2607 都经 CUDA Graph 包装（`make_rwkv7(use_graph=True)`
默认；faster3a 自身默认 graph）。warmup=20 iters=100：

| Case | faster3a_2607 | tl-fp16 | 加速比 |
|---|---|---|---|
| 1x1 decode | 18.19 ms | 23.65 ms | 0.77x（落后） |
| 1x8 | 64.51 ms | 57.32 ms | 1.13x |
| 1x32 | 166.10 ms | 71.63 ms | 2.32x |
| 1x128 | 215.33 ms | 212.67 ms | 1.01x |
| 1x256 | 484.47 ms | 401.62 ms | 1.21x |

结论：
- **decode 落后（0.77x）**：0.4b 每层 tmix_decode 拆 10 个 kernel，24 层 =
  ~264 个 kernel 顺序 launch，launch 开销主导；faster3a 用单 fusion CUDA op
  （`add_layer_norm_tmix_mix6_f16`）融整条 TMIX。下一步应把 tmix_decode 的
  rkv/rank/gate/LN 进一步融合减少 kernel 数。
- **prefill 中 T 大幅领先（1x32 快 2.32x）**：fused cmix_prefill + 融合 prologue
  的 GEMM 路径优于 faster3a 的逐 op 调度。
- **大 T 趋平**：GEMM 计算主导。

对比 0.1b：decode 领先 1.68x（C=768 小、kernel 少时融合收益 > launch 开销）；
0.4b C=1024 大时 launch 开销超过融合收益，decode 转负。prefill 优势在两种
规模都成立。
