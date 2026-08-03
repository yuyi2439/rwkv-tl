# TODO: verify compiled-prefill performance on a native-bf16 GPU

Status: RESOLVED on RTX 3060 (sm_86) -- see docs/benchmark_rwkv7_experiments.md.
This file is kept as a record; items are resolved and archived.

## 1. Does compiling `forward_prefill` beat eager?

- RESOLVED. `make_TMIX_batch` was wired to the custom ops (`fused_rwkv_gemm`,
  `fused_dplr`) and `forward_prefill` compiled cleanly (graph breaks 3 -> 0).
  Steady-state on 0.1B: faster at every T (1.43x at T=8 ... 1.11x at T=256),
  but each distinct prompt length recompiles a fresh graph (T=256 took ~702 s
  with the GPU idle). Decision: keep `forward_prefill` eager; the code was
  reverted to raw-kernel dispatch. Revisit only if graphs get cached per T or
  prompt lengths are fixed.
- RTX 3060 eager prefill for reference: rwkv_tl ~83-110 ms/32tok (0.1B),
  pure_torch ~90-101 ms/32tok.

## 2. `Tensor.item()` graph break at `forward` (single-token dispatch)

- RESOLVED/confirmed. `forward()` `int(tok.item())` (__init__.py line ~292)
  graph-breaks under dynamo (graphs=2/breaks=1, `capture_scalar_outputs=False`).
  No impact in the normal flow because `forward` is not wrapped by
  `maybe_torch_compile`; only matters if `forward` itself is compiled.
- Also observed: running `torch._dynamo.explain(m.forward)` before the
  `_run_one_impl` cache is warmed recurses inside the `maybe_torch_compile`
  wrapper (RecursionError). A real first call caches the impl; no runtime issue.

## 3. Eager vs pure-torch baseline on RTX 3060

- RESOLVED. script/bench_decode.py (0.1B, eager, in-place reset):
  rwkv_tl decode ~1344 ms/32tok vs pure_torch ~435 ms/32tok;
  rwkv_tl prefill ~83 ms/32tok vs pure_torch ~90 ms/32tok.
  Decode stays slower (small model, per-token dispatch); fused-GEMM prefill
  edges out pure_torch. Full case tables in script/benchmark_rwkv7.md.

## 4. MX450 thermal throttling skews absolute latency

- Historical note only. Validation moved to RTX 3060; MX450 numbers are
  reference-only. RTX 3060 runs measured in a single process, prefer ratios.
