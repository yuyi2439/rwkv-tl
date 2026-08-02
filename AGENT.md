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

This project is scheduled for validation on an RTX 3060 tomorrow. MX450 results are useful for relative comparisons, but they are not the final target numbers. Do not treat MX450-only measurements as the final answer. MX450 also thermal-throttles under sustained load (SM clock ~1800MHz -> ~1155MHz), inflating absolute latencies by up to ~50%; compare ratios within a single run and record clocks (see TODO.md section 4).

## Performance work

- Decode path: fused TMIX/CMIX kernels plus GraphDecoder.
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and forward_prefill must use independent state objects in tests.

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
- `maybe_torch_compile` is a plain decorator (`@maybe_torch_compile`) applied to `run_one`. The device is unknown at class-definition time, so the wrapper resolves `self.emb.device.type` lazily on the first call per instance and caches the compiled callable in `self._run_one_impl`. `torch.compile(fullgraph=True)` requires the decode path to be a single graph, so `make_TMIX`/`make_CMIX` dispatch through the registered custom ops (`torch.ops.rwkv_tl.*`) when `supports_native_bf16`; the raw method is kept as `self._eager_run_one = self.run_one.__wrapped__.__get__(self, type(self))`. The prefill path (`forward_prefill`) is not compiled and may keep raw kernels.
- Custom-op dispatch overhead is ~0.3-2 ms per call (measured `fused_lerp6_rkv_copy` at 2.2 ms/call on MX450). Using them unconditionally slowed eager decode ~10x (113 ms/token vs 66 ms/token). `make_TMIX`/`make_CMIX` take a `use_custom_ops` flag: they dispatch through `torch.ops.rwkv_tl.*` ONLY when torch.compile is enabled (`supports_native_bf16`), and call the raw tilelang kernels when eager. Never hard-code custom ops into the eager path.
- Eager perf on MX450 / 0.1B: rwkv_tl decode ~66 ms/token and prefill ~503 ms/32tok are SLOWER than pure_torch (~52 ms/token, ~285 ms/32tok). The fused kernels do not pay off on this tiny model / sm_75 GPU; expect gains only on larger models or sm_80+ (native bf16, tensor cores). Use `script/bench_decode.py` to re-measure.
- `supports_native_bf16` takes only `device_type` (no device index): `torch.cuda.is_bf16_supported()` is per-process and does not accept a device argument, and bf16 support is uniform across same-arch GPUs.

## Open verification (TODO.md)

Compiled-prefill performance and the `forward` `Tensor.item()` graph break
cannot be tested on MX450 (sm_75 inductor refuses bf16 codegen). Verify on an
sm_80+ NVIDIA GPU or AMD MI, save results, then delete `TODO.md`.

## Reference
[docs](/docs/)
