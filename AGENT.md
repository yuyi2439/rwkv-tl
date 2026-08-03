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

This is a practical compromise: the benchmark report should stay easy to skim, while the deeper notes can live in the docs and agent guide.

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
- Keep correctness first: forward and forward_prefill must use independent state objects in tests.
- `forward_prefill` is deliberately NOT torch.compile'd: each distinct prompt length recompiles a fresh graph (T=256 took ~12 min on 0.1B with the GPU idle) for a steady-state gain of only 1.11-1.43x. Keep it eager.
- The benchmark harness (`benchmark_rwkv7.py`) routes rwkv_tl/pure_torch through `_eager_run_one`/`_eager_forward_prefill` (via `_eager_dispatch`) so a sweep measures the eager implementation and never triggers per-case torch.compile recompiles (which previously made it look frozen for minutes).

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
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied to `run_one`. The device is unknown at class-definition time, so the wrapper resolves `self.emb.device.type` lazily on the first call per instance and caches the compiled callable under `self._{fn.__name__}_impl` (i.e. `run_one` -> `_run_one_impl`). `torch.compile(fullgraph=True)` requires the decode path to be a single graph, so `make_TMIX`/`make_CMIX` dispatch through the registered custom ops (`torch.ops.rwkv_tl.*`) when `supports_native_bf16`; the raw method is kept as `self._eager_run_one = self.run_one.__wrapped__.__get__(self, type(self))`. The prefill path (`forward_prefill`) is NOT compiled and keeps raw kernels (see docs/benchmark_rwkv7_experiments.md for the RTX 3060 measurement that led to this decision).
- Custom-op dispatch overhead is ~0.3-2 ms per call (measured `fused_lerp6_rkv_copy` at 2.2 ms/call on MX450). Using them unconditionally slowed eager decode ~10x (113 ms/token vs 66 ms/token). `make_TMIX`/`make_CMIX` take a `use_custom_ops` flag: they dispatch through `torch.ops.rwkv_tl.*` ONLY when torch.compile is enabled (`supports_native_bf16`), and call the raw tilelang kernels when eager. Never hard-code custom ops into the eager path.
- Eager perf on MX450 / 0.1B: rwkv_tl decode ~66 ms/token and prefill ~503 ms/32tok are SLOWER than pure_torch (~52 ms/token, ~285 ms/32tok). The fused kernels do not pay off on this tiny model / sm_75 GPU; expect gains only on larger models or sm_80+ (native bf16, tensor cores). Use `script/bench_decode.py` to re-measure.
- `supports_native_bf16` takes only `device_type` (no device index): `torch.cuda.is_bf16_supported()` is per-process and does not accept a device argument, and bf16 support is uniform across same-arch GPUs.
- The repo checkpoints are saved on cuda: `torch.load` loads them to cuda regardless of the device context, so a CPU model run needs a CPU-materialized checkpoint (`torch.load(ckpt, map_location="cpu")` then re-save).

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
- **Make benchmarks correctness-gated.** A benchmark run must verify numerical
  correctness before reporting latency. If outputs do not match the reference,
  skip the case (or fail) and do NOT emit a latency number. This prevents
  silently reporting speed for broken code. **DONE 2026-08**: added to
  `script/benchmark_rwkv7.py` (per-case, `--no-correctness-check` /
  `--correctness-tol`) and `script/bench_decode.py`; the reference is pure_torch.

## Open verification (TODO.md)

Resolved on RTX 3060 (sm_86): compiled-prefill perf and the `forward`
`Tensor.item()` graph break are now verified and documented. The compiled
prefill was measured faster (1.11-1.43x) but kept eager due to per-length
recompile cost; see TODO.md (now archived) and docs/benchmark_rwkv7_experiments.md.

## Reference
[docs](/docs/)
