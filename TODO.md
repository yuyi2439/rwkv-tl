# TODO

按优先级排列。完成一项就删掉对应条目。
进度与已关闭的死路见 `docs/progress.md`。

## P0 — decode 性能与可用性（最高优先级）

### #1 batch decode（B>1）

目前 decode 只支持 B=1。RNN 架构无 KV cache 内存爆炸，对 batching 有天然优势。
改动：State tensor 加 B 维度；`fused_dplr` grid 加 batch 维度；GEMV 变 batched GEMV
（`[B,C]×[C,C]`）。这让项目从 benchmark 工具变成可用推理引擎。batch 化同时
天然提升 GEMV 的 M 维并行度，是绕开单行 GEMV occupancy 瓶颈的正路。

### #2 头投影 GEMV（decode 第一热点 ~27%）

`[C]×[65536,C]` 每步 ~2ms，是 decode 最大单项。W8A16 手写 GEMV 已上（1.96x，
见 docs/progress.md）。进一步方向：vocab 分块 + CUDA Graph 内多 kernel 流水；
或 CUTLASS int8 GEMV 路径直接吃 2x 字节收益。

### #3 tmix rkvWt/oWt 换 W8A16 kernel

tmix 的 rkvWt/oWt 目前走 dequant→fp16（int8 释放，未常驻）。参考 cmix decode
融合 Q8 的做法，在 tmix decode 里让 rkvWt/oWt int8 常驻显存。注意：现有
[K,M] lane-per-output GEMV 受 occupancy 限制，int8 带宽收益有限（见
docs/progress.md 死路记录），先以显存收益为主。

## P1 — prefill 与大模型验证

### #4 chunk 并行 prefill

T=128 prefill 落后 faster3a 2.5x。faster3a 的 `wkv_fp16_seq_v2` 也是串行 over T
但用 (B,H) grid + fp16 register state + cp.async 双缓冲，我们已对齐 register state
（e0da4c7）。剩余差距在 front 的 4 个 rank 一阶 GEMM + rkv GEMM（faster3a 用
cuBLAS batched）。真 chunk 并行（FlashRWKV `chunk_rwkv7` / FLA 风格）是唯一
未试路径；注意 SM120 上 chunk 可能比 recurrent 慢，需实测交叉点。

**警示**：在 MX450 上调优的 kernel 必须在 sm_80+ 复测（9e81fd1 曾在 3060
倒退 2.5-10x）。

### #5 1.5B 模型验证（RTX 3060）

1.5B 已加载验证（decode 正确、prefill T=8/32 快 faster3a ~2x，decode 18.93ms
与 T≥64 prefill 落后）。后续 kernel 改动都应在 1.5B + 3060 上回归。

## P2 — 融合与训练路径

### #6 GroupNorm 融合（单层 11.3%）

`_impl_kernel_3`(GN) 与 DPLR kernel 同为 (H,N) 组织，GN 依赖 DPLR 输出 y 的
全 head 归约，现拆两 kernel 多一轮 y 读写。融合或 cooperative 归约可省一次
launch + 读写。低秩 gate 链已融合完成（fused_gates ~5.3x）。

### #7 FFN 分策略（fp16 binned / bf16 4-stream）

官方 FFN 按 dtype 分策略：fp16 用 `cmix_sparse_binned`（6-bin）+ finalize
利用稀疏性；bf16 用 4 stream 并行 split + 归约。我们 FFN 统一路径，
`[C]×[ffn_rows,C]` 大 GEMV 是 decode 另一瓶颈。

### #8 DPLR backward

长期训练支持。per-op custom op + `register_autograd`。DPLR 时间反向递推是
RWKV 最难部分，需保存每步 `S_new`（或 `S@A`/`V⊗K`），memory footprint 大，
可参考 Mamba `selective_scan` backward 的 checkpoint 策略。其他 op backward
标准，可先用 PyTorch autograd 兜底。
