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
optimization alone (chunk-parallel DPLR is not enough either; see
`docs/runs/rtx3060.md` "三模型 vs faster3a 完整差距分析"). This is an
established conclusion, not a new finding; benchmark reports state data only
and do not extrapolate to overall superiority. The project's differentiation
is quantization (TODO #6), the operator library, and ease of use.

## Guidance

- Decode path: fused TMIX/CMIX kernels; `CUDAGraph` (default via
  `make_rwkv7(..., use_graph=True)`) accelerates decode + per-T prefill with
  CUDA-Graph replay.
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and prefill must use independent state
  objects in tests.
- `prefill` is deliberately NOT torch.compile'd: each distinct prompt length
  recompiles a fresh graph (T=256 took ~12 min on 0.1B with the GPU idle) for
  a steady-state gain of only 1.11-1.43x. Keep it eager.
- The benchmark harness (`benchmark_rwkv7.py`) builds rwkv_tl/pure_torch with
  `is_torch_compile=False` and routes them through `decode`/`prefill` (via
  `_eager_dispatch`) so a sweep measures the eager implementation and never
  triggers per-case torch.compile recompiles (which previously made it look
  frozen for minutes). The correctness gate is OFF by default
  (`--correctness-check` opt-in) to keep VRAM low on 2GB GPUs. The
  `graph_decoder` benchmark target was removed when CUDA-Graph moved into the
  model wrapper; every tl target gets graphs via the `CUDAGraph` wrapper.
- Use the real scripts in `script/` rather than ad-hoc snippets.
- On memory-constrained GPUs, split large sweeps into separate processes; a
  single process can accumulate compile-cache pressure and trigger OOMs.
- History: on MX450 (sm_75) the 0.1B prefill used to beat faster3a across the
  board (tuned variants removed; historical numbers in git history). RTX 3060
  experiments are recorded in `docs/runs/rtx3060.md`.
- Compiled-prefill perf and the `forward` `Tensor.item()` graph break were
  verified on the RTX 3060 (`docs/runs/rtx3060.md`). Compiled prefill was
  faster (1.11-1.43x) but kept eager due to per-length recompile cost.
