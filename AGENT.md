# AGENT.md

Working notes for agents working on this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It is the canonical, human-facing project standard (repository layout, kernel-writing reference, conventions). AGENT.md only adds agent-specific operating rules on top of it.

## Management rules for AGENT.md

This file is the operating guide for future agents. Follow these rules strictly:

- Keep it concise and practical. Prefer short bullets over long prose.
- Keep the content in English unless a specific repository report is intentionally written in Chinese.
- Update it when a new constraint, bug, environment limitation, or workflow rule is discovered.
- Do not leave important findings only in chat history; record them here when they affect future work.
- **Documenting project standards is itself a project standard.** Whenever a
  decision establishes a lasting constraint or convention (package layout,
  interface contracts, device/dtype strategy, workflow rules), write it down
  here in the same session, before moving on -- future agents follow this
  file, not chat history. "I'll remember" is not an acceptable substitute.
- Keep this file focused on actionable guidance. Avoid personal notes, speculation, or long retrospective writing.
- When a new benchmark or experiment note is added, make sure the relevant link and summary are also reflected here.
- Update this file when a new constraint, bug, or performance path is discovered. Keep it concise and actionable.
- **After a large external/pulled refactor, verify the code against this file's claims line-by-line before trusting the commit message.** Code behavior and AGENT.md/README descriptions can silently diverge in the same commit (e.g. `make_rwkv7` `"auto"` mapping changed while AGENT.md still described the old one). Diff the actual branches, not the narrative.
- **Before cross-linking per-GPU docs, confirm the machines actually match.** A claim like "see validation_rtx3060.md (same machine)" was wrong -- MX450 (laptop, 2GB) and RTX 3060 (desktop, 12GB) are different machines. Verify hardware before asserting a shared test record.

## Management rules for benchmark_rwkv7.md

This file is the canonical benchmark report for this repository. Follow these rules strictly:

- Keep it in Chinese.
- Keep it concise and report-like. It should only contain the benchmark entry script, environment, measured results, and short explanations that directly interpret those results.
- Do not put exploratory findings, long reasoning, speculative conclusions, or operational caveats in this file.
- Put extra experimental findings in [docs/benchmarks/rtx3060.md](docs/benchmarks/rtx3060.md).
- Put runtime warnings, environment constraints, and maintenance guidance in this file.
- When a new benchmark run is added, update this report with the new numbers and keep the narrative short.

## Management rules for test validation records

- Test results (which model versions passed the test suite, environment, commit) go in a per-GPU validation file under `docs/` (e.g. `docs/validation_<gpu>.md`). Keep it in Chinese and report-like.

This is a practical compromise: the benchmark report should stay easy to skim, while the deeper notes can live in the docs and agent guide.

## User preferences and project standards (remember these)

- Docs and reports under `docs/` and the benchmark report must be written in Chinese. Source code comments/docstrings stay in English.
- Test results must be saved to a file under `docs/`. Do not leave test outcomes only in chat history.
- When a new benchmark/test run is completed, record the results in the docs before moving on.
- Model checkpoints are located via the `RWKV_CHECKPOINT_PATH` env var / `--project-checkpoint` flag (the run command in `script/benchmark_rwkv7.md` shows the exact names used); the directory is machine-specific. Tested checkpoints: rwkv7-g1d-0.1b, rwkv7-g1d-0.4b. Test the originally-used model first, then the others; watch ou for OOM.
- `prefill` stays eager: torch.compile of prefill recompiles a fresh graph per distinct prompt length (minutes, GPU idle) for only 1.11-1.43x steady-state. This was validated on RTX 3060 and is a firm decision -- do not re-enable without new evidence.
- Long benchmarks must run as background processes writing to a log file, then be monitored -- never as a blocking foreground command that looks frozen.
- If a script appears to hang with idle CPU/GPU, investigate before assuming it failed: torch.compile or first-call kernel compilation can idle the GPU for minutes.
- When the user says "check it yourself" or "you can do more tests", investigate and run any additional worthwhile tests autonomously.
- **Never run git write/state-changing operations on your own** (no `git add`, `git commit`, `git reset`, `git restore`, `git rm`, etc.). Reading state via git (`git status`, `git diff`, `git log`, `git show`, `git fetch`) is always allowed. The ONLY state-changing git operation permitted without approval is renaming/moving an already-tracked file (`git mv`). Permission for any other git write (commit/push/amend) is ALWAYS temporary and scoped to that single action; each commit/push needs its own explicit approval. When in doubt, ask.
- **Report incompatibilities; do not fix design choices on your own.** When you hit an incompatibility in user-authored code (dtype/API mismatches, a crash you can repro), STOP and tell the user directly with a repro, instead of silently changing their design (e.g. rewriting `out` dtype or adding conversions). Fixing genuine bugs (undefined behavior, crashes) is fine, but prefer flagging + suggesting the one-line fix and let the user decide. This rule came from the `generate` `out`-dtype episode.
- **Ask before design decisions.** Before proposing/implementing an architecture change (new params, new weight-storage schemes, refactors touching `weight.py` layout), present the plan and ask the user to confirm -- they have strong opinions about naming and where logic lives (e.g. "no computation-time terms like `gemm` in `weight.py`", fp32 handled in the model class not the weight). Confirm scope + naming before writing code.
- **Distill reusable lessons into AGENT.md; do not keep a running "mistakes" log.** An already-fixed code bug is not worth recording -- it will not recur, and a fresh diagnosis is fast. Only record a mistake when it teaches a reusable lesson (a workflow rule, an API pitfall, a doc-accuracy check, real friction) and write that lesson into the relevant AGENT.md section in the same session.

## Project structure and standards

These are firm, user-approved conventions. Follow them when adding or moving code.

- **`src/rwkv_tl/` is the published library; `demo/` is not part of it.**
  The package must be self-contained: no docstring or comment inside
  `src/rwkv_tl/` may reference `demo/`, `script/`, or `docs/` (files that do
  not ship with the package). Model implementations are helper/example code
  and live OUTSIDE the package in `demo/`; do not move them into `src/rwkv_tl/`.
- **Model interface.** `demo/_rwkv7_abc.py` defines the `RWKV7Model` ABC
  (`decode` / `prefill` / `forward` / `generate`). Every model implements it.
  Application scripts (`script/rwkv_chat.py`, `script/profile_prefill.py`, ...)
  build models via `demo.make_rwkv7(w, backend="auto")` and operate on the
  ABC; do not hard-code a specific model class into an application script.
- **Kernels are split by function AND bound by IO dtype.** The kernel
  definitions live in `kernels/{gemm,lerp,gates,dplr}.py`, each exposing
  `build(DTYPE)`; `kernels/_base.py` is just the assembler
  (`build_kernels(DTYPE) -> Kernels`); `kernels/fp16.py` / `kernels/bf16.py`
  bind the two dtypes with identical public interfaces; `kernels/__init__.py`
  re-exports the fp16 bindings by default. Add a new kernel in its function
  file and expose it through BOTH bindings, never one dtype file only.
  Custom ops (`operators`) route by input tensor dtype via `_kernels_for`.
- **Tilelang closure-dtype gotcha.** `kernels/{gemm,lerp,gates,dplr}.py`
  deliberately have NO `from __future__ import annotations`: tilelang's eager
  builder evaluates the annotation expressions, and a stringified annotation
  only has module globals + direct nonlocals available, so a closure `DTYPE`
  param fails with `NameError`. The pre-dtype-split files used literal
  `"float16"`/`"bfloat16"` strings and could keep the import; the `build(DTYPE)`
  files cannot.
- **Dtype plumbing.** `RWKV7Weight(path, dtype=...)` controls weight precision
  (default `torch.float16`, converts the bf16 checkpoint once at load; pass
  `torch.bfloat16` to keep the raw dtype). `State(..., dtype=...)` must match
  the model dtype. DPLR RNN state is always fp32 in both variants.
- **`demo.make_rwkv7` backends**: `"auto"` (`RWKV7FP16` on CUDA sm < 80,
  `RWKV7BF16` otherwise -- including sm >= 80 and non-CUDA devices), `"fp16"`,
  `"bf16"`, `"mx450"`, `"rtx3060"`, `"tuned"`, `"torch"`. `use_graph=True`
  (default) makes `make_rwkv7` return a class pre-wrapped in `CUDAGraph` for
  every CUDA backend, so `decode` and per-T `prefill` run from captured CUDA
  Graphs. `RWKV7Torch` updates its state in place, so it captures too; pass
  `use_graph=False` to keep a truly eager class (e.g. the torch reference
  used for correctness gating).
- **`demo.cuda_graph.CUDAGraph` is THE CUDA-Graph mechanism** (merges the old
  `demo/graph_decode.py` `GraphDecoder` + `demo/prefill_graph.py` `PrefillGraph`).
  Wrap any `RWKV7Model` instance: `model = CUDAGraph(RWKV7MX450(w))`, or via
  `wrap_model(model)`, or `make_rwkv7(..., use_graph=True)` (returns a
  pre-wrapped class). It lazily captures the wrapped model's OWN `decode`
  (T=1) and `prefill` per exact T (T<=`prefill_graph_max_t`, default 64) by
  calling them against a fixed-address shadow `State`, then copies the caller's
  `State` in/out around each replay. Larger T, non-CUDA models, and any capture
  failure fall back to the wrapped model's eager path.
- **Device-tuned variants live in `demo/tuned/` and are EAGER.** The only
  remaining dedicated variant is `RWKV7MX450` (Turing sm_75: fp32 batch
  GEMMs). `RWKV7RTX3060` was deleted: its only code was `state["x"].copy_()`
  closures, now moved into the base -- the base fp16 is graph-capturable, so
  on Ampere+ `tuned`/`rtx3060` select `RWKV7FP16` wrapped in `CUDAGraph`.
  The wrapper is applied from the outside.
  `backend="tuned"` picks one by CUDA device name and falls back to `"auto"`.
  The fallback must be a real `try/except` (selector failure -> `None` -> `"auto"`),
  not `try/finally` -- `finally` does NOT swallow exceptions, it only assigns
  `backend`, and the traceback still propagates. Non-CUDA devices are guarded in
  `make_tuned_model` (returns `None` before touching `torch.cuda`).
  Measured on RTX 3060 / 0.1B: the graph-wrapped fp16 (`tl-rtx3060`) leads
  faster3a_2607 at every T
  <= 64 (2.30ms 1x1, 2.90ms 1x8, 3.19ms 1x32, 3.81ms 1x64); T=128 stays eager
  and trails faster3a (~20 vs 7.8ms) -- large-T prefill is the common
  tilelang-path bottleneck, out of scope here.
- **CUDA-Graph prefill requires in-place `state["x"]`.** The batch closures
  must `state["x"].copy_(x[-1])`, NOT rebind `state["x"] = x[-1]`, or a
  captured graph silently corrupts state across replays (measured rnn max_abs
  1.4 vs 0.0). The fp16 base (`_rwkv7_base.py`) rebinds, so the tuned variants
  override `make_TMIX_batch`/`make_CMIX_batch` to use `copy_`. `CUDAGraph`
  detects a rebinding model (state tensor `data_ptr`s move during warmup) and
  transparently falls back to eager for the affected op.
- **Models are stateless; `State` is passed explicitly.** `State` and model are
  decoupled: models never own runtime state, and `decode`/`prefill`/`forward`/
  `generate` take a `State` argument and return it. State creation is a
  standalone helper (the benchmark's `make_state`), not a model method. The
  `CUDAGraph` wrapper preserves this: the captured graph replays against its
  own fixed-address shadow state and `decode`/`prefill` copy the caller's
  `State` in/out around each replay, so any `State` works and the model stays
  stateless. Do not add an owned-state API or a `zero_state` model method.

## Goal

Implement and validate faster RWKV7 inference paths in this repo. Keep the implementation correct and verify it with the real benchmark and test scripts.

**Long-term direction: rwkv-tl must support TRAINING.** All new operators/optimizations must keep autograd compatibility in mind:
- The registered custom ops (`torch.ops.rwkv_tl.*`) exist precisely to enable future `register_autograd` backward definitions (the standard PyTorch op dispatch path).
- CUDA Graph (`CUDAGraph` in `demo/cuda_graph.py`) is INFERENCE-ONLY by design: it captures the forward launch sequence and does not rebuild an autograd graph (replay does not record gradients, fixed buffers conflict with autograd's dynamic graph). Do not route anything training-relevant through it. `make_rwkv7(..., use_graph=True)` (default) integrates it as the `decode`/`prefill` path via a stateless copy-in/out around a fixed shadow state.
- A fully-fused single kernel is NOT inherently inference-only (unlike CUDA Graph) -- any custom CUDA kernel, fused or not, needs an explicit backward to support training. But fusing a whole layer makes training hard: you must hand-write the layer's backward (including the serial DPLR recurrence, which reverses in time and needs every intermediate state saved) and manually stage/save the per-op intermediates that autograd would otherwise keep. That is far more work and error-prone than the per-op custom-op path, where each op registers its own backward and intermediates stay in the autograd graph automatically. So: prefer per-op custom ops for training; do not build a whole-layer fused kernel for the training path.
- Measured on RTX 3060 / 0.1B: CUDA-Graph decode (via `CUDAGraph`) is already 1.63 ms/token with launch gaps squeezed to ~0.08 ms (GPU kernel time ~1.55 ms). Fusing all decode layers into one kernel would gain <0.1 ms over that and (as above) hurt training. The remaining real cost is the ~1.5 ms of GEMV compute itself.

## Core constraints

- Do not add compatibility shims; edit the implementation directly.
- Verify TileLang and PyTorch APIs before using them.
- Prefer existing project code over new helpers.
- Do not swallow exceptions. Only catch errors when recovery is meaningful.
- Keep new docstrings in src/rwkv_tl short and in English.
- Do not create extra files unless they are clearly necessary.

## Hardware note

Validation completed on an RTX 3060 (sm_86, 12GB). MX450 (sm_75, 2GB) is now an
ACTIVE optimization target (`tl-mx450`), not just historical: it is a Turing
card with pathological fp16 cuBLAS GEMMs and severe thermal throttling under
sustained load (latencies inflate up to ~50%, p90 >> p10) -- treat single-session
relative comparisons as reliable, absolute numbers as noisy.

## Performance work

- Decode path: fused TMIX/CMIX kernels; `CUDAGraph` (default via
  `make_rwkv7(..., use_graph=True)`) accelerates decode + per-T prefill with
  CUDA-Graph replay (see the stateless design above).
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and prefill must use independent state objects in tests.
- `prefill` is deliberately NOT torch.compile'd: each distinct prompt length recompiles a fresh graph (T=256 took ~12 min on 0.1B with the GPU idle) for a steady-state gain of only 1.11-1.43x. Keep it eager.
- The benchmark harness (`benchmark_rwkv7.py`) builds rwkv_tl/pure_torch with `is_torch_compile=False` and routes them through `decode`/`prefill` (via `_eager_dispatch`) so a sweep measures the eager implementation and never triggers per-case torch.compile recompiles (which previously made it look frozen for minutes). The correctness gate is OFF by default (`--correctness-check` opt-in) to keep VRAM low on 2GB GPUs. The `graph_decoder` benchmark target was removed when CUDA-Graph moved into the tuned variants; the tuned targets (`tl-mx450`/`tl-rtx3060`/`tl-tuned`) now get graphs via the `CUDAGraph` wrapper.
- MX450 tuning (0.1B) now partially beats the sm75-adapted faster3a_2607: a stable 8.2ms T=1 decode (CUDA Graph) and T>=16 prefill wins; faster3a still leads T=2/4/8. See `README.md`, `script/benchmark_rwkv7.md`, and the kernel-level analysis in `docs/benchmarks/mx450_sm75.md` (RTX 3060 experiments in `docs/benchmarks/rtx3060.md`).

## Benchmarks

Use the real scripts in script/ rather than ad-hoc snippets.

```bash
# From the repo root
.venv/bin/python -m pytest test/ -v
.venv/bin/python script/benchmark_rwkv7.py --device cuda ...
```

On memory-constrained GPUs, split large sweeps into separate processes. A single process can accumulate too much compile-cache pressure and trigger OOMs.

## Known issues and notes

- The pure-torch baseline was improved by batched prefill work.
- A token-shift aliasing bug existed in the old TMIX path. Any state update that overwrites previous state must happen only after all reads from the old state are complete.
- The benchmark harness should skip per-case OOMs rather than abort the whole sweep.
- DPLR A term must be the L2-normalized key (kk/||kk||), not raw kk. Passing raw kk silently corrupts the state update and destabilizes the recurrence (decode/prefill diverged ~14 in logits and argmax flipped). `fused_l2norm_neg_kk_a` returns `(kk_norm, B)` for this reason.
- `T.Kernel` defaults to 128 threads; reduce kernels (l2norm/dplr/gn_rkrk) only need one logical warp. The extra warps run `__shfl_xor_sync(0xffffffff, ...)` while sitting on the `threadIdx.x < 32` guard's off-path, which is UB and produces rare non-deterministic results that amplify through the recurrence. Use `threads=WARP` (defined in `_common.py`, 32 on both NVIDIA and AMD) for warp-reduce kernels.
- AMD note: tilelang's `warp_reduce_sum` keeps 32-lane logical-warp semantics on HIP (CDNA wave64 and RDNA wave32) -- see `src/tl_templates/hip/reduce.h`. Do NOT set WARP to the AMD hardware wavefront (64): the reduce would then cover only lanes 0-31 and silently drop half the reduction. `SERIAL = HEAD_DIM // WARP` stays 2 on every backend.
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied to `decode`. Whether it compiles is decided per-instance via `self._is_torch_compile` (constructor param `is_torch_compile`); if False the method runs eagerly. When compiling, the first call caches the compiled callable under `self._{fn.__name__}_impl` (i.e. `decode` -> `_decode_impl`). `torch.compile(fullgraph=True)` requires the decode path to be a single graph, so `make_TMIX`/`make_CMIX` dispatch through the registered custom ops (`torch.ops.rwkv_tl.*`) whenever `is_torch_compile=True`; eager instances (is_torch_compile=False) keep raw kernels. The prefill path (`prefill`) is NOT compiled and keeps raw kernels (see docs/benchmarks/rtx3060.md for the RTX 3060 measurement that led to this decision).
- Custom-op dispatch overhead is ~0.3-2 ms per call (measured `fused_lerp6_rkv_copy` at 2.2 ms/call on MX450). Using them unconditionally slowed eager decode ~10x (113 ms/token vs 66 ms/token). `make_TMIX`/`make_CMIX` take a `use_custom_ops` flag: they dispatch through `torch.ops.rwkv_tl.*` ONLY when torch.compile is enabled (`use_custom_ops = is_torch_compile`), and call the raw tilelang kernels when eager. Never hard-code custom ops into the eager path.
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
  ~2175 kernels regardless of T), and `CUDAGraph` (demo/cuda_graph.py)
  captures the whole prefill per exact T (no padding -- the DPLR recurrence
  advances state per token), replaying with a State copy-in/out so the model
  stays stateless. Graph beats eager at EVERY T, not just small T: measured
  on RTX 3060 / 0.1B, T=128 graph 5.72 vs eager 23.06ms, T=256 graph 9.39ms,
  T=1024 graph 32.27ms (all ~4x faster than eager). The old `T<=64` cap was
  wrong (it made T=128 prefill 17.7ms, slower than faster3a's 7.6ms); it was
  based on a "graph memory scales with T" worry that did not hold. Requires
  the batch closures to update `state["x"]` in place (`copy_`, not rebind) so
  addresses stay fixed.
- **faster3a_2607's prefill is fast WITHOUT a graph**: every op is a fused
  CUDA kernel (`wkv_seq` runs the T-dim serial DPLR in one kernel,
  `wkv_seq_grid2d` switches to a 2D-grid variant for large T via a (B,T)
  tuning table), and CUDA-Graph is only an extra launch-cost layer on top (it
  captures the whole `forward` for any BxT with no T cap, `bench_case`).
  Our eager prefill is a per-layer Python loop over many small kernels, so
  without the graph it is launch-bound. The graph closes most of that gap;
  further gains need fusing the eager prefill ops (TMIX/CMIX GEMMs and gates
  are the launch-heavy part; `fused_dplr_T` is already single-kernel).
- **Turing sm_75 fp16 GEMM is T-specialized.** `kernels/gemm.py` compiles a
  per-length tilelang kernel (native m16n8k8 MMA, 16x32x32/3-stage, autotuned on
  MX450) for fp16 on sm_75, because a dynamic-T version cannot reach that
  config's speed there (~12x slower). Lengths are restricted to `1..16` exact
  plus powers of two `32..16384`; `fused_rkv_gemm` binary-searches the smallest
  covering length, pads the input, runs, and slices back (measured fastest on
  MX450 -- kernel time scales ~linearly, larger kernels only add pad waste).
  Each distinct (C, length) compiles once lazily (~8 s on MX450) and is cached
  by tilelang, so arbitrary prompt lengths pay a one-time compile. bf16 has no
  sm_75 MMA atom -> stays on cuBLAS bmm (fast fp32 emulation there); sm_80+
  keeps the dynamic-T kernel. The dtype check in `fused_rkv_gemm` protects
  RWKV7MX450's fp32-input bmm path.
- **One prim_func can contain MULTIPLE `with T.Kernel` blocks -- each becomes a
  separate `__global__` kernel, launched in order.** Verified 2026-08-07: two
  sequential `with T.Kernel` with DIFFERENT grid/thread shapes in one
  `@T.prim_func` compile and run correctly (generated source has two
  `__global__` with different `__launch_bounds__`). The earlier "one prim_func =
  one kernel / full fusion forces a single shared 1D grid" claim was WRONG: an
  up-GEMM (2D grid over hidden 4C columns) and a down-GEMM (2D grid over C
  columns) can each keep their own optimal grid as two `with T.Kernel` blocks in
  ONE prim_func. The `[T,4C]` intermediate between them still round-trips
  GLOBAL memory (shared/registers do not survive across the two launches), so
  this is performance-equivalent to two separate jit kernels (measured N=128:
  0.090 vs 0.092 ms) -- it only packages two launches into one compiled unit /
  host call. Allocate the intermediate internally with `T.alloc_global`
  (verified). The original 1D-grid full-fusion kernel was ~20x slow not because
  "one kernel has one grid" but because that kernel was written with a single
  1D grid serializing the hidden width; keep it as a reference only.
- **tilelang `T.gemm` is NOT inherently slow on sm_80+ -- a 1D grid is.** The
  earlier "tilelang GEMM can't beat cuBLAS" claim was wrong: it came from a 1D
  grid (grid over token rows only) that serializes the whole output width in one
  block and under-parallelizes the GPU. With a 2D grid (rows x output columns)
  and a tuned tile, tilelang GEMMs MATCH cuBLAS on the RTX 3060 (0.1B, C=768,
  fp16): pure `[N,768]x[768,3072]` GEMM parity with cuBLAS; fusing the relu2
  epilogue into the up-GEMM beats the eager relu+square path ~1.1-2.4x (small N
  largest); fused up+relu2 + down+residual as two 2D kernels lands ~parity to
  1.06x vs the eager 5-kernel CMIX. Measured 2026-08-06, tuned tile
  BM=32/BN=128/BK=32/threads=128/stages=2 (`kernels/neo/cmix.py`).
- **`generate(stop=...)` matches exact token-id sequences.** This is fragile for substring stops like `"\n\nUser:"`: the model can emit that text with a different tokenization than `tokenizer.encode` (measured: model emits `[..., 28329("…。\n"), 11("\n"), 24281("User"), 59(":")]` vs `encode("\n\nUser:") == [261, 24281, 59]`), so the tail never equals the stop sequence and generation leaks the next turn. Do NOT rely on substring-text stops. Once RWKV checkpoints ship a dedicated conversation-stop token id, use THAT as the stop -- token-exact matching is then correct. (Text-based matching was considered and intentionally not implemented; the dedicated stop token supersedes it.)
- Default inference is **fp16** (checkpoints are bf16, converted once at
  `RWKV7Weight` load when `dtype=torch.float16`, the default). The bf16 path
  (`RWKV7BF16` + `RWKV7Weight(..., dtype=torch.bfloat16)`) keeps the raw
  checkpoint dtype and is a reference/experimental variant. The compile
  decision is purely `is_torch_compile`.
- `RWKV7Weight(model_path, device=None)` loads directly to the target device via `torch.load(..., map_location=device)` -- the repo checkpoints are saved on cuda, so without `device` the tensors land on cuda regardless of context. The benchmark loads ONE fresh `RWKV7Weight` per target and frees it (`del` + `gc.collect()` + `empty_cache()`) before the next target, so only one weight copy is resident at a time (MX450 has 2GB VRAM); the correctness reference shares the target's weight object.
- Kernels are compiled PER-MODEL with static H/C (model constants baked at compile time; only T_LEN stays dynamic): each kernel file exposes `@functools.cache` factories (`_dplr_kernel(H)`, `_lerp6_kernel(C)`, ...) and the wrappers dispatch by input shape. Only compile-time model constants go static; per-call sizes (token count) stay dynamic. Caveat: a factory param used ONLY in a type annotation (not the kernel body) is not captured by tilelang's `get_func_nonlocals` and causes `NameError` -- reference it in the body too (e.g. `H * N` in annotations is fine since H is used in the body).
- Compute is **fp16 IO + fp32 accumulation**, DPLR state is **fp32** (matching Albatross): RNN state `[H,N,N]` is fp32 (not fp16-rounded each step), IO (r/w/k/v) stays fp16. Weights are converted bf16->fp16 once at `RWKV7Weight` load. `_dplr_kernel`/`_dplr_T_kernel` read/write fp32 S; pure_torch reference matches.
- Prefill DPLR is a **single-shot kernel** (`fused_dplr_T` / `_dplr_T_kernel`): one launch processes the whole [T,H,N] sequence, serial state recurrence inside each (h,v_n) block. Verified: y outputs bit-match the reference through long T. Known tilelang quirk: the STORED S_out comes out fp16-rounded despite being declared fp32 (y, computed from the fp32 local, is exact); treat state precision as fp16-level across calls for now.

## Planned architecture direction

These are user-approved future directions (inspired by FlashRWKV). They are NOT
done yet. When working on the related area, remind the user whether to proceed.

- **PENDING — measure bf16 vs fp16 on RTX 3060.** Decide which precision the
  base should use on Ampere+: benchmark `tl-bf16` vs `tl-fp16` (both now
  graph-wrapped via `make_rwkv7(use_graph=True)`) on the RTX 3060 before
  settling the default. Requires the RTX 3060 box (not this MX450 laptop).

- **DONE — model code moved out of `src/rwkv_tl/`.** The library ships only
  kernels + operators + state/sampling/weight/tokenizer; model implementations
  live in `demo/` (one class per device/kernel strategy):
  `demo.rwkv7_fp16.RWKV7FP16` (tilelang fp16, sm_80+), `demo.rwkv7_bf16.RWKV7BF16`
  (tilelang bf16), `demo.rwkv7_torch.RWKV7Torch` (pure torch reference).
  Device-tuned variants live in `demo/tuned/`:
  `demo.tuned.rwkv7_mx450.RWKV7MX450`
  (sm_75: fp32 batch GEMMs + T<=16 tilelang fp16 rkv; eager, graph-safe).
  `RWKV7RTX3060` was deleted (its `copy_` closures moved into the base).
  `demo.cuda_graph.CUDAGraph` (merges the old `GraphDecoder`/`PrefillGraph`)
  wraps any `RWKV7Model` instance and provides CUDA-Graph decode + per-T
  prefill.
  `demo.make_rwkv7(device, backend=...)` returns a model *class* and picks by
  CUDA device name (`"tuned"`, default) or arch (`"auto"`).
- **Adopt a stateless operator API.** Future kernel/operator APIs should take
  `initial_state` and return `final_state` explicitly instead of mutating an
  in-place `state` dict. This is clearer, autograd-friendly, and matches the
  FlashRWKV `rwkv7(..., initial_state=, output_final_state=)` contract. Apply
  this pattern to new ops; migrate existing ones when refactoring.
- **Future: real batch (B>1) support.** Currently there is NO real batching:
  `State` has no batch dim (one `State` = one sequence), `forward` silently
  flattens `[B,T]` to a single `[B*T]` sequence (the benchmark's `BxT` cases
  are just longer single-sequence lengths, B is not real), `decode` handles
  one token, and `CUDAGraph` (like the old `GraphDecoder`) is inherently B=1. Real batch needs a
  dedicated adaptation: `State` gains a batch dim (`rnn [B,H,N,N]`, `x [B,C]`),
  prefill/decode run the batch in parallel, and the DPLR recurrence iterates
  B independent per-sequence states. User-approved direction -- DO NOT start
  this now; revisit when the user asks.

## Verified on RTX 3060 (sm_86)

- Compiled-prefill perf and the `forward` `Tensor.item()` graph break were
  verified on the RTX 3060 and are documented in
  `docs/benchmarks/rtx3060.md`. Compiled prefill was faster
  (1.11-1.43x) but kept eager due to per-length recompile cost.

## Reference
[docs](/docs/)
