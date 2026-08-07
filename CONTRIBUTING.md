# Contributing to rwkv-tl

A short guide for human contributors. For agent-specific operating rules, see
[AGENT.md](AGENT.md); this file is the canonical project standard for people.

## Repository layout

- `src/rwkv_tl/` — the published library. Kernel code lives in `src/rwkv_tl/kernels/`
  (one file per concern: `lerp.py`, `gates.py`, `dplr.py`, `gemm.py`, ...), with a
  unified export in `kernels/__init__.py`. Custom-op wrappers live in
  `src/rwkv_tl/operators/`.
- `demo/` — example model code and tuned per-GPU implementations. Not part of the
  published library.
- `script/` — benchmark and profiling scripts.
- `test/` — test cases. `script/` is for runnable scripts, not tests.
- `docs/` — benchmark reports and per-GPU validation notes (Chinese).

## Writing TileLang kernels

TileLang kernels in this project fuse the elementwise chains and GEMM/GEMV steps
of rwkv-tl. When you write or modify a kernel, **use the TileLang
example suite as the primary reference** — the `examples/` directory of [tilelang](https://github.com/tile-ai/tilelang)
covers some pattern this project relies on:

- `examples/gemm/` — `T.gemm`, autotune, persistent kernels, intrinsics. The
  `example_gemm_intrinsics.py` and `example_gemm_advanced_autotune.py` files are
  the most useful references for the `fused_rkv_gemm` TensorCore path.
- `examples/gemv/` — GEMV tiling, relevant for decode-path kernels.
- `examples/elementwise/` — fusion patterns used by `lerp.py`, `gates.py`.
- `examples/reduction/` and the `warp_reduce` / `pipeline` examples — needed for
  the DPLR state-reduction and warp-level reductions in `dplr.py`.
- `examples/dynamic_shape/` — `T.dynamic("C")` / `T.dynamic("H")` parameterization,
  which all kernels in this project use to support 0.1B and 0.4B models.

Prefer copying a working example structure (block dims, `T.alloc_fragment`
usage, `T.gemm` invocation, reduction idiom) over inventing a new pattern. When in
doubt about a TileLang API, search the examples first — they track the installed
TileLang version more reliably than memory.

## Conventions

- Docstrings in `src/rwkv_tl/` are concise English; no `Callers` sections.
- Kernels are parameterized with `T.dynamic("C")` / `T.dynamic("H")` /
  `T.dynamic("T_LEN")` — never hard-code model dimensions.
- Weight pre-stacking (e.g. `rWt_stack = torch.stack([rWt, kWt, vWt])`) is done
  outside runtime closures.
- Fused kernels must come with a unit test under `test/` (e.g. `test_fused_lerp.py`).
- Numerical consistency: bit-exact where possible; for reduction/gate kernels,
  fp32 accumulation with bf16 writeback is acceptable (efficiency over
  bit-exactness, per AGENT.md).

## Tests and benchmarks

- `pytest test/` for correctness (kernel bit-exactness, forward consistency).
- `script/benchmark_rwkv7.py` for performance; results go in
  `script/benchmark_rwkv7.md` (Chinese, report-style).
- Long benchmarks must run as background processes writing to a log file.

