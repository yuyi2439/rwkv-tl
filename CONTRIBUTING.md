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
  - `core/` — low-level/inference modules (`model.py` `RWKV7Model` token
    contract, `state.py`, `tokenizer.py`, `weight.py`, `cuda_graph.py`
    `CUDAGraph`); core must not reference code outside `core/`.
  - `text_model.py` — the upper-layer `RWKV7TextModel`: COMPOSES a
    `RWKV7Model` as `self.model` (no inheritance) + tokenizer, and adds
    `tokenize` / `detokenize` / `generate(str, ...) -> str` (with a `stop`
    string) / `chat(messages, ...) -> str` (renders
    `asset/rwkv_chat_template_v20260805.jinja`).
  - `rwkv7_tl.py` — fused tilelang model; `rwkv7_torch.py` — pure-PyTorch
    reference; `sampling.py`.
- `script/` — chat, benchmark, and profiling scripts.
- `examples/` — runnable usage examples.
- `test/` — correctness and API tests.
- `docs/` — benchmark reports and tuning notes (Chinese).

## Writing TileLang kernels

TileLang kernels in this project fuse the elementwise chains and GEMM/GEMV steps
of rwkv-tl. When you write or modify a kernel, **use the TileLang
example suite as the primary reference** — the `examples/` directory of [tilelang](https://github.com/tile-ai/tilelang)
(set `TILELANG_SRC` to your tilelang checkout, or use a pip-installed copy):

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

## Building models

The library is decoupled: create the weight, build the model, then wrap the
text layer (nothing is auto-detected; `backend` is always explicit):

```python
import rwkv_tl
from rwkv_tl.core import RWKV7Weight

w = RWKV7Weight("model.pth", device="cuda")   # weight; device/dtype fixed here
model = rwkv_tl.rwkv7_model(w, backend="tl")  # token-level RWKV7Model (CUDA-Graph by default)
text = rwkv_tl.RWKV7TextModel(model)          # text-level API
# or the one-call form:
text = rwkv_tl.rwkv7(w, backend="tl")
```

`backend` is `"tl"` (tilelang, CUDA) or `"torch"` (pure-PyTorch,
CPU-capable); `use_graph=False` disables CUDA-Graph wrapping.

## Tests and benchmarks

- Code style: the project uses [ruff](https://docs.astral.sh/ruff/) — run
  `ruff check src/ test/ script/` and `ruff format src/ test/ script/` before
  submitting changes.
- `pytest test/` for correctness (kernel bit-exactness, forward consistency,
  user API). CPU-only tests cover the text API via the pure-torch backend.
- `script/benchmark_rwkv7.py` for performance measurements; raw benchmark
  tables are not kept under `docs/`. Every rwkv_tl target is built through
  `rwkv7_model(w, backend=...)` with `is_torch_compile=False` and measured on
  `decode`/`prefill` directly, so a sweep measures the eager implementation
  and never triggers per-case torch.compile recompiles (which can look frozen
  for minutes). CUDA-Graph wrapping is on by default (`use_graph=True`);
  pass `use_graph=False` for the eager reference. The correctness gate is
  opt-in (`--correctness-check`) to keep VRAM low on 2GB GPUs. On
  memory-constrained GPUs, split large sweeps into separate processes (a
  single process can accumulate compile-cache pressure and trigger OOMs).
- Long benchmarks must run as background processes writing to a log file.
