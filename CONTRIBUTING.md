# Contributing to rwkv-tl

A short guide for human contributors. For agent-specific operating rules, see
[AGENTS.md](AGENTS.md); this file is the canonical project standard for people.

## Repository layout

- `src/rwkv_tl/` — the published library:
  - `kernel/` — TileLang operator factories. Public API is the weight-bound
    factories (`ln_kernel`, `gemv_kernel`, `cmix_decode_kernel`,
    `tmix_decode_kernel`, ...): each takes hyperparameters **and weights** at
    construction and returns a callable that only takes activations/state.
    Raw `@tilelang.jit` factories and shared macros (`gemv_macro`,
    `gemv_main_macro`, `ln_prologue_macro`, ...) stay available for custom
    fused chains. Legacy per-op kernels live in `kernel/old/`.
  - `model.py` — the stateless `RWKV7Model` interface plus the text-level API
    (`generate` / `logits` / `tune_state` / `encode` / `detokenize`).
  - `rwkv7_tl.py` — fused tilelang model; `rwkv7_torch.py` — pure-PyTorch
    reference; `cuda_graph.py` — CUDA-Graph wrapper; `sampling.py`,
    `tokenizer.py` (vocab packaged inside the package, e.g.
    `rwkv_vocab_v20230424.txt`), `state.py` (save/load), `weight.py`.
- `script/` — chat, benchmark, and profiling scripts.
- `test/` — correctness and API tests.
- `docs/` — benchmark reports and tuning notes (Chinese).

## Writing TileLang kernels

TileLang kernels in this project fuse the elementwise chains and GEMM/GEMV steps
of rwkv-tl. When you write or modify a kernel, **use the TileLang
example suite as the primary reference** — the `examples/` directory of [tilelang](https://github.com/tile-ai/tilelang)
is also checked out at `/home/yuyi2439/tilelang`:

- `examples/gemm/` — `T.gemm`, autotune, persistent kernels, intrinsics.
- `examples/gemv/` — GEMV tiling, relevant for decode-path kernels.
- `examples/elementwise/` — fusion patterns used by the cmix/tmix prologues.
- `examples/reduction/` — warp-level reductions used by the DPLR/group-norm
  kernels.

Prefer copying a working example structure (block dims, `T.alloc_fragment`
usage, `T.gemm` invocation, reduction idiom) over inventing a new pattern.

## Conventions

- Docstrings in `src/rwkv_tl/` are concise English; no `Callers` sections.
- H and C are compile-time fixed (closure variables in each `@tilelang.jit`
  kernel); distinct model sizes compile separate kernels cached by
  `@tilelang.jit`. Only `LEN` / `T_LEN` use `T.dynamic` for sequence-length
  flexibility.
- **Kernels are weight-bound factories, not raw-call conveniences.** New
  public operators expose a `*_kernel(C, DTYPE, ..., **weights)` factory that
  returns a `BoundKernel` (see `kernel/_bound.py`); callers never pass weights
  per call.
- Weight pre-stacking (e.g. `rkvWt = torch.stack([rWt, kWt, vWt])`) is done
  in `weight.py` at load time.
- Models are stateless: `State` is passed in and returned; never store runtime
  state on a model instance.
- Fused kernels must come with a unit test under `test/`.
- Numerical consistency: bit-exact where possible; for reduction/gate kernels,
  fp32 accumulation with bf16 writeback is acceptable (efficiency over
  bit-exactness, per AGENTS.md).

## Tests and benchmarks

- Code style: the project uses [ruff](https://docs.astral.sh/ruff/) — run
  `ruff check src/ test/ script/` and `ruff format src/ test/ script/` before
  submitting changes.
- `pytest test/` for correctness (kernel bit-exactness, forward consistency,
  user API). CPU-only tests cover the text API via the pure-torch backend.
- `script/check_torch_vs_official.py` cross-checks `RWKV7Torch` logits
  against the official RWKV-LM v7 demo (requires CUDA and a checkpoint).
- `script/benchmark_rwkv7.py` for performance; results go in
  `script/benchmark_rwkv7.md` (Chinese, report-style).
- Long benchmarks must run as background processes writing to a log file.
