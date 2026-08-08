# TODO

优先级从高到低。完成一项就删掉对应条目。

（#3 避免 bf16 GEMM、#4 kernel 迁移到 @tilelang.jit 均已完成；当前无未完成条目。）

---

以下条目来自官方 `RWKV7-1.5B-20260805/inference`（`~/rwkv/official-tl`）的对比分析。
官方实现是 **decode 专精**（纯 GEMV、无 batch/prefill 入口），我们的优势在 prefill
（批处理 GEMM + 单 kernel 串行 DPLR）。下列条目旨在补齐 decode 路径的短板。

## #5 打包 RKV 单 kernel（decode 减少 launch）

官方 `_build_rkv_program` 把 R/K/V 三个 GEMV + W/A/G/V 四个低秩 GEMV 压成单 kernel：
6 个输入向量（xr/xk/xv/xw/xa/xg）× 1 个打包权重 `[3C+rank_sum, C]`，
kernel 内用 `T.if_then_else` 按行段（r_end/k_end/v_end/w_end/...）选输入。
我们现在是 `fused_lerp6_rkv_copy`（RKV）+ 独立低秩 GEMV，decode 多一次 launch。

**进度（2026-08-08）**：低秩 first-step 已打包为 `fused_rank_gemv`
（`[v1t;w1t;a1t;g1t] @ [xv;xw;xa;xg]` 单 kernel，替换 4 个 cuBLAS fp16 GEMV，
Turing 上它们病态慢）；RKV 三个 GEMV 本身仍走 `fused_rkv_gemm`（T=1 特化 kernel），
未并入。decode eager GPU 实测 ~1.08x（0.4B/MX450）。

## #6 T=1 专精 WKV

官方有两个 WKV 变体：通用 `_wkv_out` 和 `_wkv_w0_t1_out`（B1T1，fused static decay bias）。
后者针对 decode 的 T=1 情形去掉分支、把 decay bias 静态融合。
我们的 `fused_dplr` 是通用版，decode 走通用路径有冗余分支。可为 T=1 加专精变体。

## #7 LayerNorm + 6 lerp 深融合

官方 `tmix_layernorm_mix6` 把 LayerNorm 和 6 个 shifted time-mix 向量（xr/xw/xk/xv/xa/xg）
压成单 kernel。我们现在是 `LAYER_NORM` + `fused_lerp6_rkv_copy` 分两步。
decode 场景 launch 敏感，合并可省一次 launch + 一次中间写回。

## #8 GroupNorm + RKV 残差 + gate 深融合

官方 `post_state` 一个 kernel 完成：3 个 `tvm_thread_allreduce`（sum/square_sum/rkv_sum）
+ GroupNorm affine + RKV 残差 + gate 终化。我们现在是 `fused_gn_rkrk` + 后续 torch 算子。

**进度（2026-08-08）**：低秩 gate 的 rank-out second-step + 激活 + v/w/a gate 数学已融合为
`fused_gates` 单 kernel（原 4 matmul + 2 激活 + 3 gate op → 1，实测快 ~5.3x）。
GroupNorm + RKV 残差 + oWt 部分尚未融合。

## #9 手写 GEMV（decode 路径，不走 TensorCore）

官方 decode 的矩阵乘全部手写：`T.vectorized(8)` 向量化加载 + `T.tvm_thread_allreduce`
归约 + `block_rows∈{1,2}` 双行优化，纯 CUDA Core FMA，不用 TensorCore。
原因：decode 是 `[C]×[C,C]`（M=1），TensorCore m16n8k16 对 M=1 利用率极低，手写 GEMV 反而更优。
我们 decode 走 `T.gemm`/cuBLAS，对 M=1 不划算。应加手写 GEMV 路径专门服务 decode。

## #10 residual add + LayerNorm + mix 融合（CMIX）

官方 `cmix_add_layernorm_mix` 把残差加 + LayerNorm + shifted channel-mix 输入压成单 kernel。
我们 CMIX 是分开的。同 #7/#8 思路，减少 launch。

## #11 data_ptr binding cache（CUDA Graph 下跳过 shape 校验）

官方 runtime 对每个 kernel 缓存 `tuple(tensor.data_ptr())`，CUDA Graph 下 tensor 地址固定，
binding 不变就跳过 shape 校验（`_validate`），减少 Python dispatch 开销。
我们 CUDA Graph 路径每次都走完整校验。可加 binding cache。

## #12 FFN 分策略（fp16 binned / bf16 4-stream）

官方 FFN 按 dtype 分策略：
- fp16：`cmix_sparse_binned`（6-bin 分桶）+ `binned_finalize`，利用 fp16 稀疏性
- bf16：4 个 CUDA stream 并行 split，每 stream 跑 `ffn_kernel` + `cmix_value_kernel`，
  最后 `cmix_finalize` 归约
我们 FFN 是统一路径。FFN 的 `[C]×[ffn_rows,C]` 大 GEMV 是 decode 的另一个瓶颈，值得分策略优化。

## #13 workspace 预分配

官方 `TileLangDecodeWorkspace` 一次性预分配所有中间量（ffn_hidden/ffn_bins/ffn_partials/
tmix_mixed/post_mixed/rank_* 等），运行时零分配。我们部分中间量仍可能触发运行时分配。
可引入 workspace 对象，构造时分配，decode/prefill 复用。
