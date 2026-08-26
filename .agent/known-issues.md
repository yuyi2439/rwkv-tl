# Known issues and notes

Read this before touching code with known bugs/regressions, or before
re-enabling a skipped test. Historical analysis lives here; the authoritative
performance conclusions are in `.agent/performance.md` (raw run records were
removed from `docs/` on 2026-08-20).

- **Fused prefill (9e81fd1) regressed on sm_86, now FIXED (3d05ebd, 493e3e6).**
  The fused prefill front/back (commit 9e81fd1) was tuned on MX450 (2.1x) but
  regressed badly on RTX 3060: 0.1B T=128 5.19 -> 30.3ms, 1.5B -> 386ms. Two
  root causes, both fixed:
  1. `_tmix_prefill_back`'s oWt projection was a serial-over-T hand-written
     GEMV (`for t in serial(LEN) for k in serial(C)`), 1704us at 0.1B/T=128 vs
     63us for batched `T.gemm` (27x); GN also serialized over T with H blocks.
     Fixed (3d05ebd): oWt as batched `T.gemm` + GN grid (LEN, H). 2.5-2.7x.
  2. `_tmix_prefill_front`'s rank-first-step flat kernel (`LEN*Rmax*4` blocks)
     exploded with large ranks: 1.5B 5.6ms/layer. Fixed (493e3e6): 4 packed
     `T.gemm` `[LEN,C]@[C,R]`. 1.5B T=128 198.7 -> 82.2ms (2.4x).
  Final (2026-08-12, RTX 3060): 0.1B T=128 11.3ms, 0.4B 31.5ms, 1.5B 82.2ms
  (vs the 9e81fd1 regression of 30.3/114.8/386.5ms; still ~2x the a8e2ef7
  baseline, remaining cost is the rank-out flat kernel). fp16 suite passes;
  greedy generate output unchanged. Remaining optimization (reported, not
  fixed): rank-out flat kernel -> packed GEMM (~4x). Preserved conclusions:
  `.agent/performance.md`.
- **Project route assessment vs faster3a_2607 (2026-08-12).** After all the
  fused-prefill fixes and DPLR/rank optimizations this session, measured on
  RTX 3060 (T=1..512 sweep): **0.1B/0.4B prefill beats faster3a** (0.1B
  T<=64, 0.4B T=8..64; 0.1B decode also wins), but **1.5B loses everywhere
  (1.3-2.1x) and all models lose large-T prefill** (T>=256, 1.5-2.0x). Two
  structural reasons faster3a wins on big models, both hard to close with
  tilelang high-level kernels alone:
  1. **decode single-row GEMV**: faster3a's `linear_orig_row1_exact_f16` uses
     128 threads + half2 vectorized-K + multi-warp reduce (41.7us on 1.5B);
     our `gemv_macro` is 32-thread lane-per-output scalar-K (94-146us in
     chain). C larger -> gap wider.
  2. **prefill GEMM scale + serial DPLR latency**: C=2048 compute is ~7x
     0.1B, run through a serial-over-T DPLR. faster3a uses cuBLAS batched +
     cp.async prefetch.
  Conclusion: rwkv-tl cannot fully surpass faster3a on large models without
  hand-written CUDA kernels (contradicting the tilelang route) or the chunk-
  parallel DPLR (TODO #3). Small models are already competitive/ahead. Full
  sweep and analysis (raw numbers removed 2026-08-20):
  `.agent/performance.md`.
- **sm_75 (MX450) sweep after the rank/DPLR optimizations (2026-08-12).** On
  this laptop 0.1B prefill is now fully ahead of faster3a: T=32 36.7 vs 49.0ms,
  T=128 56.2 vs 87.3ms (halved from 111ms after 9e81fd1), T=512 193 vs 256ms,
  T=1024 379 vs 483ms; decode 1x1 ~8.2ms ties. The only regression is T=8
  (34.3 vs 28.7ms, small-T cost of the packed rank T.gemm) -- worth a look but
  minor. So the "cannot surpass faster3a" conclusion only holds for 1.5B large-
  T prefill on sm_86; 0.1B/0.4B are ahead on both GPUs.
- **tilelang `T.Pipelined` cannot pipeline the serial-over-T DPLR data loads.**
  Attempted to prefetch the per-token k-dim inputs (kk_norm/w/B/rkv) into
  double-buffered shared to hide memory latency (faster3a uses cp.async
  prefetch). Automatic mode (`num_stages=2`) compiles + passes numerically but
  is 6.4x SLOWER on sm_75 (synchronous shared copies + pipeline-sync overhead
  with no cp.async). Manual `order/stage` (5 copies as stage-0 async producers,
  recurrence as stage-1 consumer; stage 0 = early producer per tilelang's
  `software_pipeline_async_stages`) compiles but BREAKS correctness (max_abs
  ~12.8) -- the reorder destroys the rnn state dependency. Pipelined is built
  for dependency-free loops (GEMM K loop); it cannot safely separate "prefetch
  data" from "keep compute order" for a stateful recurrence. DPLR stays
  serial-over-T reading global (L1 suffices on sm_75).
- **DPLR register-resident state is the real win (e0da4c7).** Holding each
  state column in per-thread `T.alloc_local((N,), "float16")` registers
  (faster3a's precision), grid (H,) with N threads (thread = state column v_n),
  no cross-thread reduce -- measured -13% on 1.5B T=256 (3060). A prior
  attempt with per-thread serial + fp32 rnn reads from global was SLOWER on
  sm_75 (12 blocks, low occupancy + global round-trip): register residency is
  the key, not the serial structure.
- **Rank GEMM tile size is decisive (9972971).** Rank widths are small
  (Rv/Rw/Ra/Rg 64/96/96/256 on 1.5B): BLOCK_N=64 is ~2x vs 128. An earlier
  BLOCK_N=32 attempt was 6x SLOWER than the flat per-rank kernel (too small for
  the tensor core). Match the tile to the rank width.
- **sm_75 has no cp.async.** tilelang gates `cp.async` lowering on
  `target_has_async_copy` (sm_80+). faster3a's local "support sm75" branch
  (`8906c84`) keeps the double-buffer structure but falls back to synchronous
  int4 copies with `cp_wait`/`commit` as no-ops -- no real async overlap on
  sm_75 (why DPLR there can't benefit from the prefetch pattern).
- **bf16 `tmix_decode` fails to compile on sm_86 (tilelang bug, upstream
  unfixed).** `test_bf16_consistent` errors with `Cannot find var remap for
  xrkv` (old reports said `xr`, before the r/k/v projections were batched) in
  `unsupported_dtype_legalize.cc`. Root cause (2026-08-13, read from tilelang
  source + repro attempts): sm_86 HAS native bf16, so the device side is fine;
  the crash is in the HOST-side `BF16StorageLegalize` (runs unconditionally in
  `host_codegen` because the host target is llvm and `CheckDataTypeSupport`
  only accepts cuda targets). `HoistGlobalBufferAllocations` moves
  `T.alloc_global` buffers (e.g. `xrkv`) into the host main block's
  `alloc_buffers`; the pass only pre-registers function params / let/bind vars
  in `var_remap_`, and the `AllocBuffer` visitor queries the map BEFORE its
  force-remap fallback, so the host-side bf16 alloc throws. Not a rwkv-tl
  logic bug, not a hardware limit; fp16 unaffected. **Decision (2026-08-13):
  `test_bf16_consistent` is temporarily skipped (pytest.mark.skip) until
  tilelang updates -- re-run it after any tilelang upgrade and remove the skip
  when it passes.** Verify upstream status before re-enabling: as of
  v0.1.13 / 2026-08-13 the only related fix is #19383 (target-less PrimFunc
  bad-optional-access), which is a different bug.
- **Decode regressed with the fused-kernel path (a8e2ef7).** Measured on RTX
  3060 / 0.1B: `tl-fp16` decode went 2.36ms (legacy per-op kernels + graph) to
  eager 3.44ms / graph 4.15ms -- graph is now *slower* than eager. Root cause
  (2026-08-10 profiling): `tmix_decode` launches ~11 sequential kernels per
  layer (prologue + r/k/v GEMVs + rank gates + L2norm + DPLR + GN + oWt GEMV +
  residual). The three r/k/v GEMVs are 60-73% of decode GPU time and inflate
  2.4-3.8x over their microbenchmark time (55/71/149us vs 17/19/63us at
  C=768/1024/2048) because each reads the previous kernel's freshly-written
  global buffer with no overlap, plus low occupancy (24-64 blocks of 32
  threads) and launch gaps. faster3a fuses r/k/v into one `row1_exact4` kernel
  at 14.8us each. The GEMV kernel itself is NOT slow (microbench ties
  cuBLAS). The `head @ ln_out` cuBLAS GEMV ([65536,C]) adds 0.3-0.8ms.
  Additionally `CUDAGraph.decode`'s per-token State copy-in/out (~36 tiny
  `copy_`) makes graph slower than eager. See the decode findings in
  `.agent/performance.md` (closed roads section in TODO.md). **Partially fixed: the r/k/v
  projections are now one batched GEMV (see `gemv_batch_macro` and the
  stacked `rkvWt`); on MX450/0.4B decode 1x1 beats faster3a (~1.24x).**
  The remaining decode gap on 1.5B is the single-row GEMV structure itself
  (see the route assessment above).
- The pure-torch baseline was improved by batched prefill work.
- A token-shift aliasing bug existed in the old TMIX path. Any state update
  that overwrites previous state must happen only after all reads from the old
  state are complete.
- The benchmark harness should skip per-case OOMs rather than abort the whole
  sweep.
- DPLR A term must be the L2-normalized key (kk/||kk||), not raw kk. Passing
  raw kk silently corrupts the state update and destabilizes the recurrence
  (decode/prefill diverged ~14 in logits and argmax flipped).
  `fused_l2norm_neg_kk_a` returns `(kk_norm, B)` for this reason.
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied
  to `decode`. Whether it compiles is decided per-instance via
  `self._is_torch_compile` (constructor param `is_torch_compile`); if False
  the method runs eagerly. When compiling, the first call caches the compiled
  callable under `self._{fn.__name__}_impl` (i.e. `decode` -> `_decode_impl`).
  The prefill path (`prefill`) is NOT compiled and keeps raw kernels (see
  `.agent/performance.md` for the RTX 3060 measurement that led to this
  decision; raw numbers removed 2026-08-20).
- The old `operator/` custom-op layer (`torch.ops.rwkv_tl.*`) is **deleted**
  (it wrapped the legacy per-op kernels). torch.compile of the fused decode
  path is not yet validated; if it graph-breaks, revisit after the fused
  kernels gain explicit autograd/`torch.library` support.
- **Non-contiguous cuBLAS operands are ~2.7x slower on Turing.** A transposed
  weight view (`W.T`, strides `(1, N)`) passed straight to `matmul`/`bmm` runs
  far slower than the contiguous copy (measured `[128,768]@[768,3072]` fp32:
  1.52 vs 0.55ms). Therefore `RWKV7Weight` stores every matrix weight already
  transposed-contiguous: attention `rWt/kWt/vWt/oWt` and FFN `kWt/vWt` are
  `[in, out]`, plus a shared stacked `rkvWt = stack([rWt,kWt,vWt])` and the
  low-rank rank-in gates in BOTH orientations (`w1` `[C,R]` + `w1t` `[R,C]`).
  `emb` is layer-normalized once at load. Closures therefore reference weight
  tensors directly (fp16 models add ~0 extra closure memory). This one fix cut
  MX450 prefill T=128 by ~40% (70.6 -> 43.4ms). Note the FFN is 4x expanded
  (`[C, C]` weights are actually `[4*C, C]`), so CMIX GEMMs are the prefill
  bottleneck, not the TMIX gate chains.
- **Prefill is CUDA-Graphed per exact T up to `prefill_graph_max_t` (default
  1024; `None` = no cap).** Small-T prefill is launch-bound (a constant
  ~2175 kernels regardless of T), and `CUDAGraph` (rwkv_tl/cuda_graph.py)
  captures the whole prefill per exact T (no padding -- the DPLR recurrence
  advances state per token), replaying with a State copy-in/out so the model
  stays stateless. Graph beats eager at EVERY T, not just small T: measured
  on RTX 3060 / 0.1B, T=128 graph 5.15 vs eager 23.06ms, T=256 (16x16) graph
  9.22ms, T=1024 graph 32.27ms (all ~4x faster than eager). The old `T<=64`
  cap was wrong (it made T=128 prefill 17.7ms, slower than faster3a's 7.6ms);
  it was based on a "graph memory scales with T" worry that did not hold.
  Requires the batch closures to update `state["x"]` in place (`copy_`, not
  rebind) so addresses stay fixed.
- **faster3a_2607's prefill is fast WITHOUT a graph**: every op is a fused
  CUDA kernel (`wkv_seq` runs the T-dim serial DPLR in one kernel,
  `wkv_seq_grid2d` switches to a 2D-grid variant for large T via a (B,T)
  tuning table), and CUDA-Graph is only an extra launch-cost layer on top (it
  captures the whole `forward` for any BxT with no T cap, `bench_case`).
  Our eager prefill is a per-layer Python loop over many small kernels, so
  without the graph it is launch-bound. The graph closes most of that gap;
  further gains need fusing the eager prefill ops (TMIX/CMIX GEMMs and gates
  are the launch-heavy part; `fused_dplr_T` is already single-kernel).
- **Turing sm_75 fp16 GEMM is T-specialized.** The legacy `kernel/old/gemm.py`
  `fused_rkv_gemm` (used by the TMIX prefill transition) compiles a
  per-length tilelang kernel (native m16n8k8 MMA, 16x32x32/3-stage, autotuned
  on MX450) for fp16 on sm_75, because a dynamic-T version cannot reach that
  config's speed there (~12x slower). Lengths are restricted to `1..16` exact
  plus powers of two `32..16384`; it binary-searches the smallest covering
  length, pads the input, runs, and slices back (measured fastest on MX450).
  Each distinct (C, length) compiles once lazily (~8 s on MX450) and is cached
  by tilelang, so arbitrary prompt lengths pay a one-time compile. bf16 has no
  sm_75 MMA atom -> stays on cuBLAS bmm (fast fp32 emulation there); sm_80+
  keeps the dynamic-T kernel.
- **`generate(stop=...)` matches exact token-id sequences.** This is fragile
  for substring stops like `"\n\nUser:"`: the model can emit that text with a
  different tokenization than `tokenizer.encode` (measured: model emits
  `[..., 28329("…。\n"), 11("\n"), 24281("User"), 59(":")]` vs
  `encode("\n\nUser:") == [261, 24281, 59]`), so the tail never equals the
  stop sequence and generation leaks the next turn. Do NOT rely on
  substring-text stops. Once RWKV checkpoints ship a dedicated
  conversation-stop token id, use THAT as the stop -- token-exact matching is
  then correct. (Text-based matching was considered and intentionally not
  implemented; the dedicated stop token supersedes it.)
- Default inference is **fp16** (checkpoints are bf16, converted once at
  `RWKV7Weight` load when `dtype=torch.float16`, the default). The bf16 path
  (`RWKV7TL` on `RWKV7Weight(..., dtype=torch.bfloat16)`) keeps the raw
  checkpoint dtype and is a reference/experimental variant (sm_80+ only; the
  fused kernels need bf16 tensor cores). The compile decision is purely
  `is_torch_compile`.
- `RWKV7Weight(model_path, device=None)` loads directly to the target device
  via `torch.load(..., map_location=device)` -- the repo checkpoints are saved
  on cuda, so without `device` the tensors land on cuda regardless of context.
  The benchmark loads ONE fresh `RWKV7Weight` per target and frees it (`del` +
  `gc.collect()` + `empty_cache()`) before the next target, so only one weight
  copy is resident at a time (MX450 has 2GB VRAM); the correctness reference
  shares the target's weight object.
- Fused kernels are compiled PER-SHAPE via `@tilelang.jit` /
  `@functools.cache` factories (`_cmix_decode_kernel(C, DTYPE)`,
  `_tmix_decode_kernel(C, DTYPE, H, Rv, Rw, Ra, Rg)`, ...); model constants
  (C, H, gate ranks) are baked at compile time, only sequence length stays
  dynamic (`LEN`/`T_LEN`). Weights stay at the model IO dtype -- **no fp32
  weight copies anywhere**.
- Compute is **fp16 IO + fp32 accumulation**, DPLR state is **fp32** (matching
  Albatross): RNN state `[H,N,N]` is fp32 (not fp16-rounded each step), IO
  (r/w/k/v) stays fp16. Weights are converted bf16->fp16 once at
  `RWKV7Weight` load. `tmix_decode`'s DPLR reads/writes fp32 S; the pure-torch
  reference matches.
- Prefill DPLR is a **single-shot kernel** (`fused_dplr_T` / `_dplr_T_kernel`):
  one launch processes the whole [T,H,N] sequence, serial state recurrence
  inside each (h,v_n) block. Verified: y outputs bit-match the reference
  through long T.
- **`v_first` is the "value residual": layer 0's v, passed LAYER-to-layer, NOT
  persisted across tokens** (2026-08-11). Official RWKV7 (RWKV-LM
  `rwkv_v7_demo.py`): `if layer_id == 0: v_first = v` else
  `v = v + (v_first - v) * sigmoid(v0 + xv@v1@v2)`. It is a per-forward
  temporary (layer 0 seeds it, later layers gate toward layer-0's same-row v);
  `State` does NOT store it. A wrong earlier reading treated it as
  "sequence-first-token v" persisted in `State` -- that made decode/prefill
  diverge (~7 logits in `test_decode_matches_prefill`) because the batched
  `_tmix_prefill_front` stored only row t=0 in a `[C]` buffer while decode used
  the current token. Fix: `_tmix_prefill_front` now takes `v_first: [LEN, C]`
  (layer-0's whole-batch v) and `first != 0` gates the whole batch;
  `decode`/`prefill` reset `v_first = None` at the start of each forward and
  pass it layer-to-layer. Keeping `v_first` out of `State` also means
  CUDA-Graph capture has no `None`-to-tensor rebind (decode/prefill graph both
  capture cleanly).
- **Batched prefill kernels are numerically equivalent to per-token decode**
  (`_tmix_prefill_front`/`_back`, `cmix_prefill`): `time_mix_batch`/
  `channel_mix_batch` use them; verified ~0.004-0.008 max-abs vs the per-token
  path on fp16 (batched prologue shifts via `x_ln[t-1]` vs decode's stored
  `prev_x`, plus per-row GEMV ordering).
