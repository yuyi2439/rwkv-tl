# TODO

按优先级排列。完成一项就删掉对应条目。

## P0 — decode 性能与可用性（最高优先级）

### #1 手写 GEMV（decode 路径，不走 TensorCore）

decode 是 `[C]×[C,C]`（M=1），TensorCore m16n8k16 对 M=1 利用率仅 1/16，
`T.gemm`/cuBLAS 对 GEMV 不划算。应加手写 GEMV 路径专门服务 decode：
`T.vectorized(8)` 向量化加载 + `T.tvm_thread_allreduce` 归约 + `block_rows∈{1,2}`
双行优化，纯 CUDA Core FMA。参考官方 `~/rwkv/official-tl/kernel.py` 的 GEMV 写法
及 `~/tilelang/examples/gemv/`。

graph_decoder 的 1.55ms GPU kernel time 里 GEMV 计算是主要成本，手写预计快 30-50%。
这是当前 decode 路径收益最大的单项优化。

### #2 batch decode（B>1）

目前 decode 只支持 B=1。RNN 架构无 KV cache 内存爆炸，对 batching 有天然优势。
改动：State tensor 加 B 维度；`fused_dplr` grid 加 batch 维度；GEMV 变 batched GEMV
（`[B,C]×[C,C]`）。这让项目从 benchmark 工具变成可用推理引擎。

## P1 — prefill 与大模型验证

### #3 chunk 并行 prefill

T=128 prefill 落后 faster3a 2.5x，根因是 faster3a 的 wkv_seq kernel 用 chunk 并行 +
cp.async 流水线，我们的单 kernel 串行 DPLR 在大 T 时计算效率不够。T-bucketing
（按 T 分桶捕获 graph）只省 launch 开销，不解决计算效率问题，仅作过渡。

真正解法是 chunk-based 并行 prefill：把 T 维切成 chunk（如 16/32），chunk 内并行
计算 GEMM，chunk 间串行递推 state。参考 FlashRWKV `chunk_rwkv7` 和 FLA 的实现。
注意 SM120 上 chunk 反而比 recurrent 慢（FlashRWKV 实测），需在我们的硬件上验证
交叉点。

### #4 1.5B 模型验证（RTX 3060）

0.1B/0.4B 太小，不足以暴露真实推理场景的问题。1.5B 是 RWKV7 主力部署尺寸
（bf16 需 ~3GB，RTX 3060 12GB 可装）。大模型会暴露：SMEM 不足（更大 N 影响
`fused_gn_rkrk`/`fused_dplr` register pressure）、occupancy 下降（H=64 时 grid 更大）、
DPLR N 维并行度变化。7.2B 需等量化支持后再测（bf16 ~14.4GB）。

**进度（2026-08-08）**：`rwkv7-g1i-1.5b-20260805-ctx16384.pth`（C=2048, H=32, N=64,
L=24）已在本机 3060 加载成功（g1i 权重键结构与 g1d 一致，`RWKV7Weight` 零改动），
decode 8-token 正确性验证通过（max_abs 0.039，argmax 一致）。完整 benchmark 待补。
详见 docs/benchmarks/rtx3060.md。

## P2 — 训练路径

### #5 DPLR backward

长期目标是训练支持，per-op custom op + `register_autograd` 是正确路线。
第一个 backward 应实现 `fused_dplr` 的反向：DPLR 的时间反向递推是 RWKV 最核心也
最难的部分。需保存每步 `S_new`（或 `S@A`/`V⊗K`），memory footprint 显著增加，
可参考 Mamba `selective_scan` backward 的 checkpoint 策略。

其他 op 的 backward（GEMM/LayerNorm/activation）是标准的，可先用 PyTorch autograd
兜底。elementwise 融合（lerp/gate/l2norm）的反向用 torch 算子组合即可。

## P3 — 后续优化

### #6 量化（融合进手写 GEMV kernel）

decode 是 memory-bound，量化直接减半 memory bandwidth。weight 用 int8/any4 存储，
kernel 内做 dequant + compute 融合；DPLR state 保持 fp32。应在 #1 手写 GEMV 完成
后在 GEMV kernel 内融合 dequant，而非单独做量化路径。参考 rwkv7-quantization
（any4 在 RTX 2080 Ti 达 114.7 tok/s）。

### #7 decode 路径深融合（减少 launch）

合并原多项官方借鉴，目标是减少 decode 的 kernel launch 数和中间写回：

- **LayerNorm + 6 lerp 深融合**：官方 `tmix_layernorm_mix6` 把 LayerNorm 和 6 个
  shifted time-mix 向量压成单 kernel。我们现在是 `LAYER_NORM` + `fused_lerp6_rkv_copy`
  分两步。
- **GroupNorm + RKV 残差 + gate 深融合**：官方 `post_state` 一个 kernel 完成 3 个
  reduce + GroupNorm affine + RKV 残差 + gate 终化。我们现在是 `fused_gn_rkrk` + 后续。
- **CMIX 残差 + LayerNorm + mix 融合**：官方 `cmix_add_layernorm_mix` 单 kernel。
- **打包 RKV + 低秩单 kernel**：官方 `_build_rkv_program` 把 6 个输入 × 1 打包权重
  压成单 kernel，`T.if_then_else` 按行段选输入。

进度（2026-08-08）：
- 低秩 first-step 已打包为 `fused_rank_gemv`（`[v1t;w1t;a1t;g1t] @ [xv;xw;xa;xg]` 单
  kernel，替换 4 个 cuBLAS fp16 GEMV，Turing 上它们病态慢）；RKV 三个 GEMV 仍走
  `fused_rkv_gemm`，未并入。
- 低秩 gate 的 rank-out second-step + 激活 + v/w/a gate 已融合为 `fused_gates` 单
  kernel（原 4 matmul + 2 激活 + 3 gate op → 1，实测快 ~5.3x）。
- GroupNorm + RKV 残差 + oWt 尚未融合。

### #8 FFN 分策略（fp16 binned / bf16 4-stream）

官方 FFN 按 dtype 分策略：fp16 用 `cmix_sparse_binned`（6-bin 分桶）+ `binned_finalize`
利用稀疏性；bf16 用 4 CUDA stream 并行 split + `cmix_finalize` 归约。我们 FFN 是统一
路径。FFN 的 `[C]×[ffn_rows,C]` 大 GEMV 是 decode 另一个瓶颈。

### #9 T=1 专精 WKV

官方有通用 `_wkv_out` 和 `_wkv_w0_t1_out`（B1T1，fused static decay bias）两个变体。
后者针对 decode T=1 去掉分支、静态融合 decay bias。我们的 `fused_dplr` 是通用版，
decode 走通用路径有冗余分支。

### #10 data_ptr binding cache（CUDA Graph 下跳过 shape 校验）

官方 runtime 缓存 `tuple(tensor.data_ptr())`，CUDA Graph 下地址固定，binding 不变就
跳过 shape 校验（`_validate`），减少 Python dispatch 开销。我们 CUDA Graph 路径每次
都走完整校验。

