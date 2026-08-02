# TODO: verify compiled-prefill performance on a native-bf16 GPU

Cannot be verified on MX450 (sm_75): `torch._inductor` refuses to codegen bf16
("NVIDIA GeForce MX450 does not support bfloat16 compilation natively"), so
torch.compile runs nothing there. Run these checks on an sm_80+ NVIDIA GPU or
an AMD MI card, save the results, then delete this file.

## 1. Does compiling `forward_prefill` beat eager?

- Current state: `maybe_torch_compile` wraps only `run_one` (decode). The prefill
  path (`forward_prefill`) is always eager.
- Measure: on an sm_80+/AMD box, compare eager vs compiled prefill using
  `script/bench_decode.py` (add a compiled-prefill case if useful). If the
  compiled prefill is faster, wrap `forward_prefill` with `maybe_torch_compile`
  too.
- Prerequisite to make prefill a single graph: `make_TMIX_batch` still calls raw
  tilelang kernels (`fused_rkv_gemm`, `fused_dplr`), which graph-break on
  `PrimExprWithOp`. Wire them to the registered custom ops
  (`torch.ops.rwkv_tl.fused_rkv_gemm` / `.fused_dplr`) with the same
  `use_custom_ops` flag pattern as `make_TMIX`. The serial DPLR loop
  (`for t in range(T_len)`) is traced by unrolling; expect a recompile per
  distinct prompt length.

## 2. `Tensor.item()` graph break at `forward` (single-token dispatch)

- `src/rwkv_tl/__init__.py` `forward()` line ~239:
  `return self.run_one(int(tok.item()), S)` graph-breaks under dynamo
  (`capture_scalar_outputs=False`).
- Impact today: none for the normal flow, because `forward` is NOT wrapped by
  `maybe_torch_compile` (only `run_one` and the eager prefill dispatch are). It
  only matters if someone compiles `forward` directly.
- If `forward` ever becomes a compile entry point, avoid the break with
  `torch._dynamo.config.capture_scalar_outputs = True`, or restructure the
  single-token path to not call `.item()` (e.g. keep a tensor token and use
  `F.embedding`, as `GraphDecoder._run_step` already does).

## 3. Eager vs pure-torch baseline on MX450 (reference numbers)

Measured with `script/bench_decode.py` on MX450 / 0.1B (eager, in-place reset):

| implementation | decode ms/32tok | prefill ms/32tok |
| --- | --- | --- |
| rwkv_tl | ~2850 | ~230 |
| pure_torch | ~1320 | ~290 |

rwkv_tl eager is slower on decode here because the tiny model + sm_75 make launch
overhead dominate; the fused-GEMM prefill path is now faster than pure_torch.
Re-measure on the target GPU (RTX 3060+ / AMD MI) to decide whether the fused
kernels + torch.compile actually win.

## 4. MX450 thermal throttling skews absolute latency

Sustained load on the MX450 laptop GPU throttles the SM clock from ~1800 MHz to
~1155 MHz (verified via nvidia-smi), inflating absolute latencies by up to ~50%
and making cross-session numbers not directly comparable. Confirmed not a code
regression: pure_torch (unchanged) slowed by the same factor, and the
rwkv_tl/pure_torch ratio stayed ~3x on T=1. When benchmarking on the target GPU,
measure all implementations in one run and prefer ratios over absolutes; record
GPU clocks alongside results.
