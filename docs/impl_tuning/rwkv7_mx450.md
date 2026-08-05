# RWKV7MX450 特调分析

对应文件: [demo/tuned/rwkv7_mx450.py](../../demo/tuned/rwkv7_mx450.py)

## 定位

针对 **Turing sm_75**（笔记本 MX450）的特调。继承 `RWKV7FP16` 的 fp16 decode 路径，
在 prefill 上做 **fp32 matrix workaround** 绕开 Turing 的 fp16 cuBLAS 病态。

## 特调点

### 1. prefill GEMM 全走 fp32（核心 workaround）

```python
rWt_stack = att.rkvWt.float()          # fp32（weight 存 fp16 stack，这里 cast 一次）
oWt = att.oWt.float()                  # fp32
# input 也 cast: x.float(), xr.float() ...
```

**原因**: Turing 上 cuBLAS 对 prefill 小形状 `[T,C]@[C,C]`（T=32..128）会选 **fp16 tensor-core kernel**，性能病态：

| 路径 | 实测 (MX450) |
|---|---|
| fp16 bmm | ~1.3ms |
| fp32 bmm | ~0.16ms |

fp16 比 fp32 **慢 4-8x**。Ampere+ 不受影响，base 类在那里保持 fp16。

### 2. 小 T (≤16) 走 tilelang fp16 m16n8k8 kernel

```python
if T_len <= 16:
    rkv = fused_rkv_gemm(xr.half(), xk.half(), xv.half(), rWt_stack16)  # tilelang fp16
else:
    rkv = fused_rkv_gemm(xr, xk, xv, rWt_stack)  # fp32 cuBLAS
```

**原因**: T 与最优 GEMM 路径存在交叉点（实测）：

| T | tilelang fp16 | fp32 cuBLAS | 胜者 |
|---|---|---|---|
| T=8 | 0.12ms | 0.21ms | tilelang |
| T=128 | 0.83ms | 0.42ms | cuBLAS |

T≤16 时 launch 开销占比大，tilelang 单 kernel 融合（r/k/v 一次算出）更划算；T≥32 后 cuBLAS 的 tiling / 软件流水线优势显现。

## 根本驱动

**硬件病态驱动**: Turing 的 fp16 cuBLAS 对 prefill 形状选错算法，必须用 fp32 绕开；同时利用 tilelang 单 kernel 融合在小 T 区间反超。这是**真硬件特性 workaround**，换硬件（sm_80+）就消失，所以特调文件会长期存在。
