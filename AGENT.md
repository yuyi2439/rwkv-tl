# AGENT.md

Working notes for agents working on this repository.

## Management rules for AGENT.md

This file is the operating guide for future agents. Follow these rules strictly:

- Keep it concise and practical. Prefer short bullets over long prose.
- Keep the content in English unless a specific repository report is intentionally written in Chinese.
- Update it when a new constraint, bug, environment limitation, or workflow rule is discovered.
- Do not leave important findings only in chat history; record them here when they affect future work.
- Keep this file focused on actionable guidance. Avoid personal notes, speculation, or long retrospective writing.
- When a new benchmark or experiment note is added, make sure the relevant link and summary are also reflected here.
- Update this file when a new constraint, bug, or performance path is discovered. Keep it concise and actionable.

## Management rules for benchmark_rwkv7.md

This file is the canonical benchmark report for this repository. Follow these rules strictly:

- Keep it in Chinese.
- Keep it concise and report-like. It should only contain the benchmark entry script, environment, measured results, and short explanations that directly interpret those results.
- Do not put exploratory findings, long reasoning, speculative conclusions, or operational caveats in this file.
- Put extra experimental findings in [docs/benchmark_rwkv7_experiments.md](docs/benchmark_rwkv7_experiments.md).
- Put runtime warnings, environment constraints, and maintenance guidance in this file.
- When a new benchmark run is added, update this report with the new numbers and keep the narrative short.

## Management rules for test validation records

- Test results (which model versions passed the test suite, environment, commit) go in [docs/validation_rtx3060.md](docs/validation_rtx3060.md) (or a sibling per-GPU file). Keep it in Chinese and report-like.

This is a practical compromise: the benchmark report should stay easy to skim, while the deeper notes can live in the docs and agent guide.

## User preferences and project standards (remember these)

- Docs and reports under `docs/` and the benchmark report must be written in Chinese. Source code comments/docstrings stay in English.
- Test results must be saved to a file under `docs/` (see `docs/validation_rtx3060.md` for the current RTX 3060 13/13 record). Do not leave test outcomes only in chat history.
- When a new benchmark/test run is completed, record the results in the docs before moving on.
- Models live in `~/rwkv/model/` (rwkv7-g1d-0.1b, rwkv7-g1d-0.4b, rwkv7-g1i-7.2b). Test the originally-used model first, then the others; the 7.2B may OOM on 12GB.
- `prefill` stays eager: torch.compile of prefill recompiles a fresh graph per distinct prompt length (minutes, GPU idle) for only 1.11-1.43x steady-state. This was validated on RTX 3060 and is a firm decision -- do not re-enable without new evidence.
- Long benchmarks must run as background processes writing to a log file, then be monitored -- never as a blocking foreground command that looks frozen.
- If a script appears to hang with idle CPU/GPU, investigate before assuming it failed: torch.compile or first-call kernel compilation can idle the GPU for minutes.
- When the user says "check it yourself" or "you can do more tests", investigate and run any additional worthwhile tests autonomously.
- Permission to run git write operations (commit/push/amend) is ALWAYS temporary and scoped to that single action. Each commit/push needs its own explicit approval; do not treat one approval as a standing green light. When in doubt, ask.

## Goal

Implement and validate faster RWKV7 inference paths in this repo. Keep the implementation correct and verify it with the real benchmark and test scripts.

## Core constraints

- Do not add compatibility shims; edit the implementation directly.
- Verify TileLang and PyTorch APIs before using them.
- Prefer existing project code over new helpers.
- Do not swallow exceptions. Only catch errors when recovery is meaningful.
- Keep new docstrings in src/rwkv_tl short and in English.
- Do not create extra files unless they are clearly necessary.

## Hardware note

Validation completed on an RTX 3060 (sm_86, 12GB). MX450 results are historical references only (that GPU also thermal-throttles under sustained load, inflating latencies up to ~50%).

## Performance work

- Decode path: fused TMIX/CMIX kernels plus GraphDecoder.
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and prefill must use independent state objects in tests.
- `prefill` is deliberately NOT torch.compile'd: each distinct prompt length recompiles a fresh graph (T=256 took ~12 min on 0.1B with the GPU idle) for a steady-state gain of only 1.11-1.43x. Keep it eager.
- The benchmark harness (`benchmark_rwkv7.py`) builds rwkv_tl/pure_torch with `is_torch_compile=False` and routes them through `decode`/`prefill` (via `_eager_dispatch`) so a sweep measures the eager implementation and never triggers per-case torch.compile recompiles (which previously made it look frozen for minutes).

## Benchmarks

Use the real scripts in script/ rather than ad-hoc snippets.

```bash
cd /home/yuyi2439/rwkv/rwkv-tl
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
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied to `decode`. Whether it compiles is decided per-instance via `self._is_torch_compile` (constructor param `is_torch_compile`); if False the method runs eagerly. When compiling, the first call caches the compiled callable under `self._{fn.__name__}_impl` (i.e. `decode` -> `_decode_impl`). `torch.compile(fullgraph=True)` requires the decode path to be a single graph, so `make_TMIX`/`make_CMIX` dispatch through the registered custom ops (`torch.ops.rwkv_tl.*`) when `supports_native_bf16`; eager instances (is_torch_compile=False) keep raw kernels. The prefill path (`prefill`) is NOT compiled and keeps raw kernels (see docs/benchmark_rwkv7_experiments.md for the RTX 3060 measurement that led to this decision).
- Custom-op dispatch overhead is ~0.3-2 ms per call (measured `fused_lerp6_rkv_copy` at 2.2 ms/call on MX450). Using them unconditionally slowed eager decode ~10x (113 ms/token vs 66 ms/token). `make_TMIX`/`make_CMIX` take a `use_custom_ops` flag: they dispatch through `torch.ops.rwkv_tl.*` ONLY when torch.compile is enabled (`supports_native_bf16`), and call the raw tilelang kernels when eager. Never hard-code custom ops into the eager path.
- RTX 3060 eager results (0.1B, 32 tok): rwkv_tl decode ~9.6 ms/tok and prefill ~17.9 ms/32tok now BEAT pure_torch (decode ~14.5, prefill ~121.8). The single-shot `fused_dplr_T` prefill kernel is the big win (flat ~15-18 ms across T). Use `script/benchmark_rwkv7.py --cases 1x1,1x32 --targets rwkv_tl,pure_torch` to re-measure (bench_decode.py was merged into it and removed).
- `supports_native_bf16` takes only `device_type` (no device index): `torch.cuda.is_bf16_supported()` is per-process and does not accept a device argument, and bf16 support is uniform across same-arch GPUs.
- The repo checkpoints are saved on cuda: `torch.load` loads them to cuda regardless of the device context, so a CPU model run needs a CPU-materialized checkpoint (`torch.load(ckpt, map_location="cpu")` then re-save).
- Kernels are compiled PER-MODEL with static H/C (model constants baked at compile time; only T_LEN stays dynamic): each kernel file exposes `@functools.cache` factories (`_dplr_kernel(H)`, `_lerp6_kernel(C)`, ...) and the wrappers dispatch by input shape. Only compile-time model constants go static; per-call sizes (token count) stay dynamic. Caveat: a factory param used ONLY in a type annotation (not the kernel body) is not captured by tilelang's `get_func_nonlocals` and causes `NameError` -- reference it in the body too (e.g. `H * N` in annotations is fine since H is used in the body).
- State is **fp32io16** (matching Albatross): RNN state `[H,N,N]` is fp32 (not bf16-rounded each step), IO (r/w/k/v) stays bf16. `_dplr_kernel`/`_dplr_T_kernel` read/write fp32 S; pure_torch reference matches.
- Prefill DPLR is a **single-shot kernel** (`fused_dplr_T` / `_dplr_T_kernel`): one launch processes the whole [T,H,N] sequence, serial state recurrence inside each (h,v_n) block. Verified: y outputs bit-match the reference through long T. Known tilelang quirk: the STORED S_out comes out bf16-rounded despite being declared fp32 (y, computed from the fp32 local, is exact); treat state precision as bf16-level across calls for now.

## Planned architecture direction

These are user-approved future directions (inspired by FlashRWKV). They are NOT
done yet. When working on the related area, remind the user whether to proceed.

- **Move `src/rwkv_tl/__init__.py` model code out of the library.** The current
  `RWKV7` class there is a temporary usage example, not the library core. It
  should be relocated to a `demo/` (or similar) directory outside `src/rwkv_tl`,
  so the library ships only kernels + operators. Do not keep growing model logic
  in `__init__.py`; treat it as example code already slated for extraction.
- **Adopt a stateless operator API.** Future kernel/operator APIs should take
  `initial_state` and return `final_state` explicitly instead of mutating an
  in-place `state` dict. This is clearer, autograd-friendly, and matches the
  FlashRWKV `rwkv7(..., initial_state=, output_final_state=)` contract. Apply
  this pattern to new ops; migrate existing ones when refactoring.

## Verified on RTX 3060 (sm_86)

- Compiled-prefill perf and the `forward` `Tensor.item()` graph break were
  verified on the RTX 3060 and are documented in
  `docs/benchmark_rwkv7_experiments.md`. Compiled prefill was faster
  (1.11-1.43x) but kept eager due to per-length recompile cost.

## Reference
[docs](/docs/)
