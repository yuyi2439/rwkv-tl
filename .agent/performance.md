# Performance work

Read this when doing performance work, benchmarking, or making performance
claims.

## Large-model constraint (user-stated 2026-08-13)

Large-model inference is NOT chasing faster3a (the agent did not discover this
itself; the user pointed it out). Measured on RTX 3060: 0.1B/0.4B prefill
(T<=128) beats faster3a, but 1.5B loses 1.3-2.1x on decode AND prefill;
decode at large C loses on the single-row GEMV structure, prefill on GEMM
scale + serial DPLR latency. Root cause: faster3a's hand-written CUDA kernels
(cp.async, row1_exact, split-K), which the tilelang route cannot close by
optimization alone (chunk-parallel DPLR is not enough either; see the
preserved RTX 3060 findings below). This is an
established conclusion, not a new finding; benchmark reports state data only
and do not extrapolate to overall superiority. The project's differentiation
is quantization (TODO #5), the operator library, and ease of use.

## RTX 3060 findings (preserved from the deleted run record)

`docs/runs/rtx3060.md` was deleted on 2026-08-20 (user-stated: old test data
no longer needed). Durable conclusions, without the benchmark tables:

- Decode 1x1 bottleneck is FUSION, not GEMV speed: one `tmix_decode` layer is
  ~11 sequential kernels; r/k/v GEMVs take 60-73% of decode GPU time and
  expand 2.4-3.8x in the chain (each reads the previous kernel's global
  buffer -> no overlap, low occupancy), while standalone tilelang GEMV matches
  cuBLAS. Direction: merge r/k/v into one batched GEMV, fuse the layer,
  specialize HEAD (TODO #2).
- Prefill must use batched GEMM, never serial per-token GEMV loops (a
  doubly-serial oWt accumulation regressed fused prefill up to ~10x on the
  3060; fixed 2.7-4.7x by batched `T.gemm` + grid-parallel GN). Full history:
  `.agent/known-issues.md`.
- Large-T prefill (T>=256) loses 1.5-2.0x on serial DPLR latency, not
  precision: fp16 register-resident DPLR state (aligned to faster3a) did not
  close the gap; faster3a hides latency with cp.async prefetch. Direction:
  chunk-parallel DPLR (TODO #3).
- 16x16 comparisons vs faster3a are invalid for rwkv_tl: rwkv_tl flattens
  B*T into one sequence, faster3a runs real batch [16,16].
- bf16 vs fp16 on sm_86: after fused gates + CUDA-Graph, bf16 decode matches
  fp16; the old "bf16 slower on sm_86" conclusion is invalid.
- tilelang GEMM trap: a 1D grid (one block per row, whole hidden width
  serialized in one block) made a fused FFN ~20x slower than eager and looked
  like "tilelang loses to cuBLAS"; with a 2D grid
  (BM32/BN128/BK32/threads128/stages2) tilelang `T.gemm` matches cuBLAS.
  Prefer 2D grids for GEMM.
- `cmix_prefill` `recompute=True` (default) wins or ties on the 3060 too
  (T=128 1.57x); no per-hardware branch needed.

## MX450 decode findings (2026-08-25, 0.1B g1d)

Attribution via single-layer tmix_decode_kernel probe + kernel-source dump
(tooling note at the end of this file). Per step: rkvWt batched GEMV
45%, gy@oWt GEMV 25%, GroupNorm 11%, gates 8.3%, DPLR state update only
4.5% (whole-decode ~1.4%).

- rkvWt GEMV (3x[768,768] fp16, [K,M] lane-per-output): THREADS sweep
  32/64/128/256 is FLAT (74.6/74.7/77.3/77.1us). 32 threads already read a
  full 128B sector; the kernel is occupancy-bound (72 blocks x 32 threads,
  ~14% of MX450's 16 SMs -> ~48GB/s, ~50% peak). The fix would be
  split-K/multi-warp-per-output, which tilelang cannot express. Do not
  re-litigate THREADS for this GEMV.
- int8 probe (per-group G=128 symmetric quant, dequant fused in the GEMV):
  compiles and is correct (rel err 1.1e-2 = quant floor) but only +8%
  (68.5us vs 74.6us fp16). Bytes halved, time unchanged -> the GEMV is
  latency-bound, so int8's bandwidth win cannot materialize on this
  structure. int8 pays off only where a kernel is truly bandwidth-bound
  (e.g. the head projection, which stays on CUTLASS torch.mv) or for VRAM
  savings.
- DPLR/DeltaLog: not worth porting (4.5% single-layer share, 1.4% whole
  decode; fp32 state removes the precision motive). Closed.
- CUDAGraph decode: bare-model decode does only 2 DtoD copies/step (0.1%) --
  the old "73 copies/step" premise was stale after v0.2. The wrapper's own
  copy-in/out was removed by capturing against the caller's State and
  snapshot/restore of the capture baseline (bitwise-equal to eager, fast
  path on address match, copy fallback otherwise); the per-step
  `token.item()` host sync was dropped in the same change.

## Guidance

- Decode path: fused TMIX/CMIX kernels; CUDA-Graph replay accelerates decode
  + per-T prefill (how to build models and run benchmarks: CONTRIBUTING.md
  "Building models" / "Tests and benchmarks").
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and prefill must use independent state
  objects in tests.
- `prefill` is deliberately NOT torch.compile'd: each distinct prompt length
  recompiles a fresh graph (T=256 took ~12 min on 0.1B with the GPU idle) for
  a steady-state gain of only 1.11-1.43x. Keep it eager.
- Use the real scripts in `script/` rather than ad-hoc snippets.
- History: on MX450 (sm_75) the 0.1B prefill used to beat faster3a across the
  board (tuned variants removed; historical numbers in git history). RTX 3060
  conclusions are preserved in the findings section above; the raw per-run
  benchmark record was deleted on 2026-08-20.
- Compiled-prefill perf and the `forward` `Tensor.item()` graph break were
  verified on the RTX 3060 (raw numbers removed 2026-08-20). Compiled prefill was
  faster (1.11-1.43x) but kept eager due to per-length recompile cost.
