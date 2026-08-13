# Kernel-writing conventions (rwkv_tl)

Read this file when writing or modifying TileLang kernels. Firm conventions;
the `tilelang-writer` skill covers the how-to knowledge (see AGENTS.md
"Skills").

## Weight-bound factories

- The operator set lives in `kernel/{cmix,tmix,gemv,ln}.py`; the PUBLIC API is
  the bound factories: each takes `(C, DTYPE, ...)` **plus the weights** at
  construction and returns a `BoundKernel` (see `kernel/_bound.py`) whose call
  only takes activations/state: `ln_pre = ln_kernel(C, DTYPE, W, B); y =
  ln_pre(x)`.
- Both granularities are exported from `kernel/__init__.py`: fine-grained
  composable ops (`ln_kernel`, `ln_per_row_kernel`, `gemv_kernel`,
  `gemv_batch_kernel`) and coarse fused layer kernels (`cmix_decode_kernel`,
  `cmix_prefill_kernel`, `tmix_decode_kernel`, `tmix_prefill_kernel`).
- The raw `@tilelang.jit` factories and shared macros (`gemv_macro`,
  `gemv_main_macro`, `ln_prologue_macro`, ...) stay available for custom fused
  chains. Legacy per-op kernels live in `kernel/old/`.
- The old split layout (`kernel/{gemm,lerp,gates,dplr}.py` dtype-split
  namespaces, plus the `operator/` custom ops) must not be reintroduced.
- No fp32 weight copies anywhere (a future quantization path must not multiply
  weight memory).
- Weight binding is at the wrapper level (weights held by reference); TileLang
  has no compile-time tensor-constant support, so the CUDA kernel still
  receives weight pointers per launch -- do not pretend otherwise in docs.

## GEMV contract

- `gemv_main_macro` computes the fp32 `acc` fragment and returns it;
  `gemv_macro` stores that `acc` to `out` as-is. To fuse a post-processing
  step (e.g. `relusq`, a residual add) into a GEMV store, use
  `gemv_main_macro` and process `acc` before storing.

## DSL rules

- No `from __future__ import annotations` in tilelang DSL files: tilelang's
  eager builder evaluates annotation expressions at build time, and a
  stringified annotation only resolves module globals + direct nonlocals, so a
  closure `DTYPE`/`C` param fails with `NameError`. The pre-dtype-split files
  used literal `"float16"`/`"bfloat16"` strings and could keep the import;
  `build(DTYPE)` files cannot.

## Docstring conventions for `src/rwkv_tl/`

- Parameter requirements go in the `Args:` section of the docstring of the
  function that takes that parameter (e.g. a macro factory's `M`/`K`
  divisibility constraints go in its own docstring, not a free-standing
  "Requires: ..." paragraph).
- Tensor layout requirements (e.g. `W` must be `[M, K]`) go in the macro
  `_impl`'s docstring next to that tensor's `Args:` entry, so editors surface
  them where the parameter is declared.
- Function-internal tuning knobs (block sizes, `VEC`, `STAGES`) are inline
  comments, not docstring material.
- `src/rwkv_tl` is a published library: docstrings/comments must not mention
  specific hardware names (e.g. `MX450`), other projects it was compared
  against (e.g. `Albatross`), or benchmark/test-environment results. Such
  measurements belong in `docs/`, not in shipped code.
- Do not restate in an `Args:` entry what the signature already shows (e.g.
  `out: T.Tensor((M,), DTYPE)` needs no `out: Output vector [M]` line); only
  add layout or semantic notes the signature cannot convey.
