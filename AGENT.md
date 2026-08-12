# AGENT.md

Working notes for agents working on this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It is the canonical, human-facing project standard (repository layout, kernel-writing reference, conventions). AGENT.md only adds agent-specific operating rules on top of it.

## Management rules for AGENT.md

This file is the operating guide for future agents. Follow these rules strictly.
(Doc-writing style rules -- what to write, what to skip, conventions -- live in
the `docs-writer` skill; read it before updating this file.)

- **Verify before writing — and re-verify after big changes.** Any rule that
  asserts something about the current code (a symbol, a default value, a file
  path, an API split) must be checked against the source before it is written
  -- grep/read the code, then write. Do not state how something "used to" work
  or how it "should" work; stale or invented details (e.g. naming a removed
  parameter) mislead readers and are worse than omitting the detail. After a
  large external/pulled refactor, go further and check this file's claims
  line-by-line against the code before trusting the commit message -- behavior
  and descriptions silently diverge in the same commit (e.g. `make_rwkv7`
  `"auto"` mapping changed while AGENT.md still described the old one). Diff
  the actual branches, not the narrative.
- Update it when a new constraint, bug, environment limitation, or workflow rule is discovered.
- Do not leave important findings only in chat history; record them here when they affect future work.
- **Proactively record project standards the user states.** When the user states content that is a project standard (a convention, a rule, a design preference for this repo), update AGENT.md in the same session — do not leave it only in chat history or ask for confirmation.
- **Self-improve skills on user feedback.** When the user raises a question or issue about behavior governed by a skill (`.agent/skill/<name>/SKILL.md`), update that skill so it prevents the problem next time.
- When a new benchmark or experiment note is added, make sure the relevant link and summary are also reflected here.
- **Before cross-linking per-GPU docs, confirm the machines actually match.** A claim like "see the RTX 3060 record (same machine)" was wrong -- MX450 (laptop, 2GB) and RTX 3060 (desktop, 12GB) are different machines. Verify hardware before asserting a shared test record.

## Management rules for benchmark_rwkv7.md

This file is the canonical benchmark report for this repository. Follow these rules strictly:

- Keep it in Chinese.
- Keep it concise and report-like. It should only contain the benchmark entry script, environment, measured results, and short explanations that directly interpret those results.
- Do not put exploratory findings, long reasoning, speculative conclusions, or operational caveats in this file.
- Put extra experimental findings in [docs/benchmarks/rtx3060.md](docs/benchmarks/rtx3060.md).
- Put runtime warnings, environment constraints, and maintenance guidance in this file.
- When a new benchmark run is added, update this report with the new numbers and keep the narrative short.

## Management rules for test validation records

- Validation is automated: run `pytest test/` against the checkpoint to verify
  correctness. Do not maintain a hand-written per-GPU validation document under
  `docs/` -- test outcomes that matter are captured by the test suite itself
  (the old `docs/validation_*.md` files were removed for this reason).
- Keep correctness gate notes (which model versions / GPUs pass) in this file
  or the benchmark report when relevant, not in a dedicated validation doc.

This is a practical compromise: the benchmark report should stay easy to skim, while the deeper notes can live in the docs and agent guide.

## User preferences and project standards (remember these)

- Docs and reports under `docs/` and the benchmark report must be written in Chinese. Source code comments/docstrings stay in English, and `src/rwkv_tl` docstrings stay short.
- Correctness validation is automated via `pytest test/` (no hand-written validation doc); record test outcomes only when they change a decision or are a notable gate, not as a routine log.
- When a new benchmark/test run is completed, record the results in the docs before moving on.
- **A refactor (module rename/move, path changes) must update every affected
  reference in code docstrings and project docs (AGENT.md, CONTRIBUTING.md,
  `script/benchmark_rwkv7.md`, `docs/`). Do not leave old path/name references
  behind just because the code still imports.** This is ordinary code hygiene,
  not doc work -- fix it in the same pass as the rename, and grep for stale
  names (e.g. the old `kernels/`/`operators` paths) after moving code.
- Model checkpoints are located via the `RWKV_CHECKPOINT_PATH` env var / `--project-checkpoint` flag (the run command in `script/benchmark_rwkv7.md` shows the exact names used); the directory is machine-specific. Tested checkpoints: rwkv7-g1d-0.1b, rwkv7-g1d-0.4b. Test the originally-used model first, then the others; watch ou for OOM.
- **Weights are never stored or duplicated above 16 bit/param.** No fp32
  weights, fp32 weight copies, or fp32-input GEMMs as a performance lever
  (2x weight VRAM). fp32 is allowed only for compute internals: fp32
  accumulation inside kernels, fp32 RNN state (`[H,N,N]`, matches Albatross),
  fp32 intermediate math. This includes dropping the old MX450 sm_75
  workaround (fp32 prefill GEMMs to dodge Turing's pathological fp16 cuBLAS
  kernel selection): solve the cuBLAS kernel-choice problem with tilelang
  hand-written kernels instead (the T-specialized sm_75 fp16 GEMM already does
  this). The planned memory-savings direction is quantization (int8/any4
  weights, dequant fused into the hand-written GEMV); see TODO #6.
- `prefill` stays eager: torch.compile of prefill recompiles a fresh graph per distinct prompt length (minutes, GPU idle) for only 1.11-1.43x steady-state. This was validated on RTX 3060 and is a firm decision -- do not re-enable without new evidence.
- Long benchmarks must run as background processes writing to a log file, then be monitored -- never as a blocking foreground command that looks frozen.
- If a script appears to hang with idle CPU/GPU, investigate before assuming it failed: torch.compile or first-call kernel compilation can idle the GPU for minutes.
- When the user says "check it yourself" or "you can do more tests", investigate and run any additional worthwhile tests autonomously.
- **Never run git write/state-changing operations on your own** (no `git add`, `git commit`, `git reset`, `git restore`, `git rm`, etc.). Reading state via git (`git status`, `git diff`, `git log`, `git show`, `git fetch`) is always allowed. The ONLY state-changing git operation permitted without approval is renaming/moving an already-tracked file (`git mv`). Permission for any other git write (commit/push/amend) is ALWAYS temporary and scoped to that single action; each commit/push needs its own explicit approval. When in doubt, ask.
- **Report incompatibilities; do not fix design choices on your own.** When you hit an incompatibility in user-authored code (dtype/API mismatches, a crash you can repro), STOP and tell the user directly with a repro, instead of silently changing their design (e.g. rewriting `out` dtype or adding conversions). Fixing genuine bugs (undefined behavior, crashes) is fine, but prefer flagging + suggesting the one-line fix and let the user decide. This rule came from the `generate` `out`-dtype episode.
- **Ask before design decisions.** Before proposing/implementing an architecture change (new params, new weight-storage schemes, refactors touching `weight.py` layout), present the plan and ask the user to confirm -- they have strong opinions about naming and where logic lives (e.g. "no computation-time terms like `gemm` in `weight.py`", fp32 handled in the model class not the weight). Confirm scope + naming before writing code.

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
- **Kernels are dtype-parameterized factories, not dtype-split bindings.** The
  fused kernel set lives in `kernel/{cmix,tmix,gemv,ln}.py`; each factory takes
  `(C, DTYPE, ...)` and returns a `@tilelang.jit` kernel (`cmix_decode(C,
  DTYPE)`, `cmix_prefill(C, DTYPE, LEN_block)`, `tmix_decode(C, DTYPE, H, Rv,
  Rw, Ra, Rg)`, `gemv(M, K, DTYPE)`), re-exported from `kernel/__init__.py`.
  The shared building-block macros (`gemv_macro`, `gemv_main_macro`,
  `gemv_batch_macro`, `ln_prologue_macro`, `ln_per_row_macro`) are also
  factory-returned and re-exported. The legacy
  per-op kernels (`fused_lerp6`, `fused_dplr_T`, ...) live in `kernel/old/`
  (bound by `build_kernels(DTYPE)`); TMIX prefill still uses them until a fused
  `tmix_prefill` exists. The old split layout (`kernel/{gemm,lerp,gates,dplr}.py`
  each holding dtype-split `fp16`/`bf16` namespaces, plus the `operator/` custom
  ops) was folded into `kernel/old/` and must not be reintroduced. No fp32
  weight copies anywhere (a future quantization path must not multiply weight
  memory).
- **GEMV: `gemv_main_macro` computes the fp32 `acc` fragment and returns it;
  `gemv_macro` stores that `acc` to `out` as-is.** To fuse a post-processing
  step (e.g. `relusq`, a residual add) into a GEMV store, use
  `gemv_main_macro` and process `acc` before storing.
- **Tilelang DSL files are a Python project standard: no `from __future__ import
  annotations`.** tilelang's eager builder evaluates annotation expressions at
  build time, and a stringified annotation only resolves module globals +
  direct nonlocals, so a closure `DTYPE`/`C` param fails with `NameError`. The
  pre-dtype-split files used literal `"float16"`/`"bfloat16"` strings and could
  keep the import; `build(DTYPE)` files cannot.
- **Docstring conventions for `src/rwkv_tl/` code.** Parameter requirements go
  in the `Args:` section of the docstring of the function that takes that
  parameter (e.g. a macro factory's `M`/`K` divisibility constraints go in its
  own docstring, not a free-standing "Requires: ..." paragraph). Tensor layout
  requirements (e.g. `W` must be `[M, K]`) go in the macro `_impl`'s docstring
  next to that tensor's `Args:` entry, so editors surface them where the
  parameter is declared. Function-internal tuning knobs (block sizes, `VEC`,
  `STAGES`) are inline comments, not docstring material.
- **`src/rwkv_tl` is a published library: docstrings/comments must not mention
  specific hardware names (e.g. `MX450`), other projects it was compared
  against (e.g. `Albatross`), or benchmark/test-environment results.** Such
  measurements belong in `docs/`, not in shipped code. Also do not restate in
  an `Args:` entry what the signature already shows (e.g. `out: T.Tensor((M,),
  DTYPE)` needs no `out: Output vector [M]` line); only add layout or semantic
  notes the signature cannot convey.
- **Dtype plumbing.** `RWKV7Weight(path, dtype=...)` controls weight precision
  (default `torch.float16`, converts the bf16 checkpoint once at load; pass
  `torch.bfloat16` to keep the raw dtype). `State(..., dtype=...)` must match
  the model dtype. DPLR RNN state is always fp32 in both variants.
- **`demo.make_rwkv7` backends**: `"auto"` (and `"fp16"`/`"bf16"`/`"tl"`/
  `"tuned"`/`"mx450"`/`"rtx3060"`) all select `RWKV7TL`; `"torch"` selects
  `RWKV7Torch`. The per-device tuned variants were folded into `RWKV7TL` once
  the fp32-GEMM cuBLAS workaround was dropped. `use_graph=True` (default) makes
  `make_rwkv7` return a class pre-wrapped in `CUDAGraph` for every CUDA
  backend, so `decode` and per-T `prefill` run from captured CUDA Graphs.
  `RWKV7Torch` updates its state in place, so it captures too; pass
  `use_graph=False` to keep a truly eager class (e.g. the torch reference
  used for correctness gating).
- **`demo.cuda_graph.CUDAGraph` is THE CUDA-Graph mechanism.** Wrap any
  `RWKV7Model` instance: `model = CUDAGraph(RWKV7TL(w))`, or via
  `wrap_model(model)`, or `make_rwkv7(..., use_graph=True)` (returns a
  pre-wrapped class). It lazily captures the wrapped model's OWN `decode`
  (T=1) and `prefill` per exact T (T<=`prefill_graph_max_t`, default 1024) by
  calling them against a fixed-address shadow `State`, then copies the caller's
  `State` in/out around each replay. Larger T, non-CUDA models, and any capture
  failure fall back to the wrapped model's eager path.
- **CUDA-Graph prefill requires in-place `state["x"]`.** The batch closures
  must `state["x"].copy_(x[-1])`, NOT rebind `state["x"] = x[-1]`, or a
  captured graph silently corrupts state across replays (measured rnn max_abs
  1.4 vs 0.0). `rwkv7_tl.time_mix*`/`channel_mix*` use `copy_`. `CUDAGraph`
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
- The fused kernels (`cmix_decode`, `tmix_decode`, ...) are plain tilelang
  kernels called from `demo.rwkv7_tl`; training support will need explicit
  backward definitions (a future `torch.library` registration path), not
  autograd through the raw kernel calls.
- CUDA Graph (`CUDAGraph` in `demo/cuda_graph.py`) is INFERENCE-ONLY by design: it captures the forward launch sequence and does not rebuild an autograd graph (replay does not record gradients, fixed buffers conflict with autograd's dynamic graph). Do not route anything training-relevant through it. `make_rwkv7(..., use_graph=True)` (default) integrates it as the `decode`/`prefill` path via a stateless copy-in/out around a fixed shadow state.
- A fully-fused single kernel is NOT inherently inference-only (unlike CUDA Graph) -- any custom CUDA kernel, fused or not, needs an explicit backward to support training. But fusing a whole layer makes training hard: you must hand-write the layer's backward (including the serial DPLR recurrence, which reverses in time and needs every intermediate state saved) and manually stage/save the per-op intermediates that autograd would otherwise keep. That is far more work and error-prone than the per-op custom-op path, where each op registers its own backward and intermediates stay in the autograd graph automatically. So: prefer per-op custom ops for training; do not build a whole-layer fused kernel for the training path.
- Measured on RTX 3060 / 0.1B: CUDA-Graph decode (via `CUDAGraph`) is already 1.63 ms/token with launch gaps squeezed to ~0.08 ms (GPU kernel time ~1.55 ms). Fusing all decode layers into one kernel would gain <0.1 ms over that and (as above) hurt training. The remaining real cost is the ~1.5 ms of GEMV compute itself.

## Core constraints

- Do not add compatibility shims; edit the implementation directly.
- Verify TileLang and PyTorch APIs before using them.
- Prefer existing project code over new helpers.
- Do not swallow exceptions. Only catch errors when recovery is meaningful.
- Do not create extra files unless they are clearly necessary.

## Skills

Project-guide and TileLang-writing knowledge lives in skills under
`.agent/skill/`, not in this file. Before writing/editing a TileLang kernel,
read `tilelang-writer`; before updating AGENT.md / adding a skill, read
`docs-writer` (placement rules, what not to write, skill conventions).
Apply the skill; do not re-derive or re-document what it already covers. Add
new hard-won TileLang findings to the skill, not to AGENT.md.

## Hardware note

Validation completed on an RTX 3060 (sm_86, 12GB). MX450 (sm_75, 2GB) is now an
ACTIVE optimization target (`tl-mx450`), not just historical: it is a Turing
card with pathological fp16 cuBLAS GEMMs and severe thermal throttling under
sustained load (latencies inflate up to ~50%, p90 >> p10) -- treat single-session
relative comparisons as reliable, absolute numbers as noisy.
**MX450 (sm_75) has no bf16 tensor cores: do NOT test or benchmark bf16 here**
(tilelang bf16 kernels can fail to compile/lower on this device, e.g.
"Cannot find var remap for <buffer>" in `StorageLegalizer`). bf16 paths must be
validated on sm_80+ (RTX 3060 box); MX450 work is fp16-only.
**Kernels tuned on MX450 must be re-verified on sm_80+.** MX450's pathological
fp16 cuBLAS makes hand-written serial kernels look good there, but sm_86
exposes under-parallelized implementations (see the 9e81fd1 prefill regression:
a serial-over-T oWt GEMV looked fine on MX450 but was 27x slower than batched
`T.gemm` on RTX 3060). Benchmark both cards before trusting an MX450-tuned
kernel for the 3060 target.

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

- **Fused prefill (9e81fd1) regresses on sm_86: serial oWt GEMV in `_tmix_prefill_back`.**
  The new fused prefill front/back (commit 9e81fd1, "prefill rkv projection via fp16
  T.gemm") was tuned on MX450 (2.1x there) but regresses badly on RTX 3060: 0.1B
  T=128 5.19 -> 30.3ms, 0.4B 14.6 -> 114.8ms, 1.5B T=128 35 -> 386ms (T-larger =
  worse, up to 10x). Root cause (2026-08-12 profiling): `_tmix_prefill_back`'s
  output projection is a hand-written GEMV with `for t in T.serial(LEN) for k in
  T.serial(C)` -- fully serial, no tensor cores, 1704us at 0.1B/T=128 vs 63us for
  the batched `T.gemm` used by the old `out_mm` (27x). The GN kernel likewise
  serializes over T with only H blocks. On MX450 the serial GEMV beat its
  pathological fp16 cuBLAS, hence the commit's claim; sm_86 exposes it. Fix: use
  batched `T.gemm` for oWt and GN (report, not fixed). See
  `docs/benchmarks/rtx3060.md` "fused prefill 接入（9e81fd1）".
- **bf16 `tmix_decode` fails to compile on sm_86 (a8e2ef7).** `test_bf16_consistent`
  errors with `Cannot find var remap for xr` in `unsupported_dtype_legalize.cc`.
  Same error family as the MX450 sm_75 note below, but reproduced on RTX 3060
  (sm_86), so it is a bug in the new `tmix_decode` bf16 path, not a hardware
  limit. fp16 is unaffected (full suite minus bf16 passes). Reproduced by
  `test_forward.py::test_bf16_consistent`.
- **Decode regressed with the neo-kernel path (a8e2ef7).** Measured on RTX 3060
  / 0.1B: `tl-fp16` decode went 2.36ms (legacy per-op kernels + graph) to
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
  `copy_`) makes graph slower than eager. See `docs/benchmarks/rtx3060.md`
  "neo kernels 基线" analysis and TODO #1. **Partially fixed: the r/k/v
  projections are now one batched GEMV (see `gemv_batch_macro` and the
  stacked `rkvWt`); on MX450/0.4B decode 1x1 beats faster3a (~1.24x).**
- The pure-torch baseline was improved by batched prefill work.
- A token-shift aliasing bug existed in the old TMIX path. Any state update that overwrites previous state must happen only after all reads from the old state are complete.
- The benchmark harness should skip per-case OOMs rather than abort the whole sweep.
- DPLR A term must be the L2-normalized key (kk/||kk||), not raw kk. Passing raw kk silently corrupts the state update and destabilizes the recurrence (decode/prefill diverged ~14 in logits and argmax flipped). `fused_l2norm_neg_kk_a` returns `(kk_norm, B)` for this reason.
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied to `decode`. Whether it compiles is decided per-instance via `self._is_torch_compile` (constructor param `is_torch_compile`); if False the method runs eagerly. When compiling, the first call caches the compiled callable under `self._{fn.__name__}_impl` (i.e. `decode` -> `_decode_impl`). The prefill path (`prefill`) is NOT compiled and keeps raw kernels (see docs/benchmarks/rtx3060.md for the RTX 3060 measurement that led to this decision).
- The old `operator/` custom-op layer (`torch.ops.rwkv_tl.*`) is **deleted** (it wrapped the legacy per-op kernels). torch.compile of the fused decode path is not yet validated; if it graph-breaks, revisit after the fused kernels gain explicit autograd/`torch.library` support.
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
  on RTX 3060 / 0.1B, T=128 graph 5.15 vs eager 23.06ms, T=256 (16x16) graph
  9.22ms, T=1024 graph 32.27ms (all ~4x faster than eager). The old `T<=64` cap was
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
- **Turing sm_75 fp16 GEMM is T-specialized.** The legacy `kernel/old/gemm.py`
  `fused_rkv_gemm` (used by the TMIX prefill transition) compiles a
  per-length tilelang kernel (native m16n8k8 MMA, 16x32x32/3-stage, autotuned on
  MX450) for fp16 on sm_75, because a dynamic-T version cannot reach that
  config's speed there (~12x slower). Lengths are restricted to `1..16` exact
  plus powers of two `32..16384`; it binary-searches the smallest covering
  length, pads the input, runs, and slices back (measured fastest on MX450).
  Each distinct (C, length) compiles once lazily (~8 s on MX450) and is cached
  by tilelang, so arbitrary prompt lengths pay a one-time compile. bf16 has no
  sm_75 MMA atom -> stays on cuBLAS bmm (fast fp32 emulation there); sm_80+
  keeps the dynamic-T kernel.
- **`generate(stop=...)` matches exact token-id sequences.** This is fragile for substring stops like `"\n\nUser:"`: the model can emit that text with a different tokenization than `tokenizer.encode` (measured: model emits `[..., 28329("…。\n"), 11("\n"), 24281("User"), 59(":")]` vs `encode("\n\nUser:") == [261, 24281, 59]`), so the tail never equals the stop sequence and generation leaks the next turn. Do NOT rely on substring-text stops. Once RWKV checkpoints ship a dedicated conversation-stop token id, use THAT as the stop -- token-exact matching is then correct. (Text-based matching was considered and intentionally not implemented; the dedicated stop token supersedes it.)
- Default inference is **fp16** (checkpoints are bf16, converted once at
  `RWKV7Weight` load when `dtype=torch.float16`, the default). The bf16 path
  (`RWKV7TL` on `RWKV7Weight(..., dtype=torch.bfloat16)`) keeps the raw
  checkpoint dtype and is a reference/experimental variant (sm_80+ only; the
  fused kernels need bf16 tensor cores). The compile decision is purely
  `is_torch_compile`.
- `RWKV7Weight(model_path, device=None)` loads directly to the target device via `torch.load(..., map_location=device)` -- the repo checkpoints are saved on cuda, so without `device` the tensors land on cuda regardless of context. The benchmark loads ONE fresh `RWKV7Weight` per target and frees it (`del` + `gc.collect()` + `empty_cache()`) before the next target, so only one weight copy is resident at a time (MX450 has 2GB VRAM); the correctness reference shares the target's weight object.
- Fused kernels are compiled PER-SHAPE via `@tilelang.jit` / `@functools.cache`
  factories (`_cmix_decode_kernel(C, DTYPE)`, `_tmix_decode_kernel(C, DTYPE,
  H, Rv, Rw, Ra, Rg)`, ...); model constants (C, H, gate ranks) are baked at
  compile time, only sequence length stays dynamic (`LEN`/`T_LEN`). Weights
  stay at the model IO dtype -- **no fp32 weight copies anywhere**.
- Compute is **fp16 IO + fp32 accumulation**, DPLR state is **fp32** (matching
  Albatross): RNN state `[H,N,N]` is fp32 (not fp16-rounded each step), IO
  (r/w/k/v) stays fp16. Weights are converted bf16->fp16 once at
  `RWKV7Weight` load. `tmix_decode`'s DPLR reads/writes fp32 S; the pure-torch
  reference matches.
- Prefill DPLR is a **single-shot kernel** (`fused_dplr_T` / `_dplr_T_kernel`): one launch processes the whole [T,H,N] sequence, serial state recurrence inside each (h,v_n) block. Verified: y outputs bit-match the reference through long T.
- **`v_first` is the "value residual": layer 0's v, passed LAYER-to-layer, NOT persisted across tokens** (2026-08-11). Official RWKV7 (RWKV-LM `rwkv_v7_demo.py`): `if layer_id == 0: v_first = v` else `v = v + (v_first - v) * sigmoid(v0 + xv@v1@v2)`. It is a per-forward temporary (layer 0 seeds it, later layers gate toward layer-0's same-row v); `State` does NOT store it. A wrong earlier reading treated it as "sequence-first-token v" persisted in `State` -- that made decode/prefill diverge (~7 logits in `test_decode_matches_prefill`) because the batched `_tmix_prefill_front` stored only row t=0 in a `[C]` buffer while decode used the current token. Fix: `_tmix_prefill_front` now takes `v_first: [LEN, C]` (layer-0's whole-batch v) and `first != 0` gates the whole batch; `decode`/`prefill` reset `v_first = None` at the start of each forward and pass it layer-to-layer. Keeping `v_first` out of `State` also means CUDA-Graph capture has no `None`-to-tensor rebind (decode/prefill graph both capture cleanly).
- **Batched prefill kernels are numerically equivalent to per-token decode** (`_tmix_prefill_front`/`_back`, `cmix_prefill`): `time_mix_batch`/`channel_mix_batch` use them; verified ~0.004-0.008 max-abs vs the per-token path on fp16 (batched prologue shifts via `x_ln[t-1]` vs decode's stored `prev_x`, plus per-row GEMV ordering).

## Planned architecture direction

These are user-approved future directions (inspired by FlashRWKV). They are NOT
done yet. When working on the related area, remind the user whether to proceed.

- **PENDING — measure bf16 vs fp16 on RTX 3060.** Decide which precision the
  base should use on Ampere+: benchmark `tl-bf16` vs `tl-fp16` (both now
  graph-wrapped via `make_rwkv7(use_graph=True)`) on the RTX 3060 before
  settling the default. Requires the RTX 3060 box (not this MX450 laptop).

- **DONE — model code lives in `demo/`, functional style.** The library ships
  only kernels + state/sampling/weight/tokenizer (in `rwkv_tl.kernel` /
  `rwkv_tl.state` / ...); model implementations live in `demo/`:
  `demo.rwkv7_tl.RWKV7TL` (fused tilelang kernels, fp16/bf16) and
  `demo.rwkv7_torch.RWKV7Torch` (pure torch reference). The model is
  functional (like `rwkv7_torch`): module-level `time_mix` / `channel_mix` /
  `time_mix_batch` / `channel_mix_batch` take the weight + input + state dict,
  and `RWKV7TL.decode`/`prefill` loop the blocks. `demo.cuda_graph.CUDAGraph`
  wraps any `RWKV7Model` instance and provides CUDA-Graph decode + per-T
  prefill. `demo.make_rwkv7(device, backend=...)` returns a model class; all
  tilelang backends resolve to `RWKV7TL` (the per-device `tuned` variants were
  deleted with the fp32-GEMM workaround).
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
