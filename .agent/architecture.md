# Architecture direction

Read this when planning architecture changes, touching the model API /
entry-point / CUDA-Graph surface, or designing new features. This file holds
the core API contract; AGENTS.md keeps only the one-line pointers.

## Core API contract

Firm user-stated API. Verify any detail against the source before writing it.

### Package layout and self-containment

- `src/rwkv_tl/` is the published library: models, the tokenizer (vocab
  packaged in the wheel), sampling, state, and CUDA-Graph all live in the
  package. No docstring/comment inside `src/rwkv_tl/` may reference `script/`
  or `docs/` (docstring conventions: `.agent/kernels.md`).
- `core/` is dependency-free: it holds only the low-level/inference modules
  (`model` / `state` / `tokenizer` / `weight` / `cuda_graph`). Code in core
  MUST NOT reference anything outside core, while external modules may import
  from core. `CUDAGraph` wraps any `RWKV7Model` (inference layer only).
- The pure-torch functional ops (`time_mix` / `channel_mix` /
  `time_mix_batch` / `channel_mix_batch`) live in `rwkv_tl.rwkv7_torch`.

### Text layer: RWKV7TextModel

- `rwkv_tl.text_model.RWKV7TextModel` (outside core) COMPOSES a token-level
  `RWKV7Model` as `self.model` (no inheritance) plus a decoupled tokenizer,
  and exposes the full inference surface by delegation (`decode` / `prefill`
  / `forward` / `w`).
- Adds `tokenize` / `detokenize` / `generate(str, decoding params,
  stop=...) -> str` / `chat(messages, ...) -> str` (renders
  `asset/rwkv_chat_template_v20260805.jinja`). `generate` takes a string
  prompt, returns only the newly generated text, and stops early when the
  output contains the `stop` string (the match is truncated away).

### Entry points and model interface

The caller owns the steps: create the `RWKV7Weight` manually
(`RWKV7Weight(path, device=..., dtype=...)`), then
`rwkv_tl.rwkv7_model(w, *, backend, use_graph=True, **kwargs)` -> `RWKV7Model`:
maps the weight to the backend token-level model (optionally CUDA-Graph
wrapped). `backend` is REQUIRED: `"tl"` (tilelang, CUDA) or `"torch"`
(pure-PyTorch, CPU-capable). There is no auto-detection --
`torch.cuda.is_available()` must never pick an implementation (user-stated
2026-08-20: some devices report CUDA but cannot actually run it).
- `rwkv_tl.rwkv7(w, *, backend, use_graph=True, **kwargs)` -> `RWKV7TextModel`:
  one-call convenience for `RWKV7TextModel(rwkv7_model(w, ...))`; the caller
  still creates the weight manually.
- Removed over time: the class-factory `make_rwkv7` (2026-08-21, superseded
  by `rwkv7_model`), the `"auto"` backend (2026-08-20), the `device` /
  `device_name` factory args (2026-08-20), the tuned variants
  (`tl-mx450` / `tl-rtx3060` / `tl-tuned`) and the dtype-carrying backend
  names (`"fp16"` / `"bf16"`, 2026-08-13). Weight precision is controlled
  exclusively by `RWKV7Weight(dtype=...)`.
- `rwkv_tl.core.model.RWKV7Model` is the token-only stateless ABC (`decode` /
  `prefill` / `forward`); `RWKV7TL` / `RWKV7Torch` are the explicit
  token-level classes, and every backend implements `RWKV7Model`. Application
  scripts build models via `rwkv7_model()` / `rwkv7()`; do not hard-code a
  specific model class. The token-class constructors take `w: RWKV7Weight`
  only (identical declaration to `RWKV7Model.__init__`); they no longer
  accept a checkpoint path or `device`/`dtype` (2026-08-21).
- All interfaces are STATELESS: `State` is passed into `decode`/`prefill`;
  models never own runtime state.

### Dtype plumbing

- `RWKV7Weight(path, dtype=...)` controls weight precision: default
  `torch.float16` converts the bf16 checkpoint once at load; pass
  `torch.bfloat16` to keep the raw dtype (reference/experimental variant,
  sm_80+ only).
- `State(..., dtype=...)` must match the model dtype. DPLR RNN state is
  always fp32 (`[H,N,N]`, matches Albatross).
- Weights are never stored or duplicated above 16 bit/param; fp32 is allowed
  only for compute internals (kernel accumulation, RNN state, intermediate
  math). The planned memory-savings direction is quantization (int8/any4
  weights, dequant fused into the hand-written GEMV); see TODO #5.

### CUDA-Graph mechanism

- `CUDAGraph` (`core/cuda_graph.py`) is THE CUDA-Graph mechanism: wrap a
  CUDA `RWKV7Model` directly (`CUDAGraph(model)`). It contains only
  CUDA-Graph logic and must never wrap a non-CUDA model.
- Conditional wrapping goes through
  `rwkv_tl.core.cuda_graph.try_cuda_graph(model, use_graph=True)` -> the
  wrapped model when the model's actual device is CUDA (no
  `torch.cuda.is_available()` probe) and `use_graph` is enabled, else the
  original model. `rwkv7_model()` / `rwkv7()` use it themselves
  (`use_graph=True` default).
- Capture is lazy (T=1 decode; per-T prefill up to `prefill_graph_max_t`,
  default 1024); capture requires in-place `state["x"]` updates (`copy_`,
  not rebind); larger T and capture failures fall back to eager.
- The wrapper is STATEFUL BY DESIGN: it owns exactly one internal `State`
  (`CUDAGraph.state`). Only calls passing that State get graph replay (zero
  state copies); any other `State` runs the wrapped model eagerly -- there is
  deliberately no copy-in/out bridge (routing foreign states through one
  graph is an aliasing hazard). Backends stay stateless and shareable; the
  graph wrapper is the one stateful singleton on top. `RWKV7TextModel.model`
  does not care whether it is graph-wrapped.
- CUDA-Graph is inference-only: replay does not build an autograd graph (see
  the training section below).

## Long-term: support TRAINING

All new operators/optimizations must keep autograd compatibility in mind:
- The fused kernels (`cmix_decode`, `tmix_decode`, ...) are plain tilelang
  kernels called from `rwkv_tl.rwkv7_tl`; training support will need explicit
  backward definitions (a future `torch.library` registration path), not
  autograd through the raw kernel calls.
- CUDA Graph (`CUDAGraph` in `rwkv_tl/core/cuda_graph.py`) is INFERENCE-ONLY by
  design: it captures the forward launch sequence and does not rebuild an
  autograd graph (replay does not record gradients, fixed buffers conflict
  with autograd's dynamic graph). Do not route anything training-relevant
  through it. `rwkv7_model(..., use_graph=True)` / `rwkv7(...)` (default)
  integrate it as the `decode`/`prefill` path via its own internal State:
  calls passing `CUDAGraph.state` replay the graph, others run eager.
- A fully-fused single kernel is NOT inherently inference-only (unlike CUDA
  Graph) -- any custom CUDA kernel, fused or not, needs an explicit backward
  to support training. But fusing a whole layer makes training hard: you must
  hand-write the layer's backward (including the serial DPLR recurrence, which
  reverses in time and needs every intermediate state saved) and manually
  stage/save the per-op intermediates that autograd would otherwise keep. That
  is far more work and error-prone than the per-op custom-op path, where each
  op registers its own backward and intermediates stay in the autograd graph
  automatically. So: prefer per-op custom ops for training; do not build a
  whole-layer fused kernel for the training path.
- Measured on RTX 3060 / 0.1B: CUDA-Graph decode (via `CUDAGraph`) is already
  1.63 ms/token with launch gaps squeezed to ~0.08 ms (GPU kernel time ~1.55
  ms). Fusing all decode layers into one kernel would gain <0.1 ms over that
  and (as above) hurt training. The remaining real cost is the ~1.5 ms of
  GEMV compute itself.

## Planned directions (user-approved, NOT done)

When working on the related area, remind the user whether to proceed.

- **PENDING — measure bf16 vs fp16 on RTX 3060.** Decide which precision the
  base should use on Ampere+: benchmark `tl-bf16` vs `tl-fp16` (both now
  graph-wrapped via `rwkv7_model(..., use_graph=True)`) on the RTX 3060 before
  settling the default. Requires the RTX 3060 box (not the MX450 laptop).

- **DONE — models live in the library (0.2 refactor).** `RWKV7TL` (fused
  tilelang kernels, fp16/bf16) and `RWKV7Torch` (pure torch reference,
  CPU-capable) are part of the published package; see "Core API contract"
  above for the current surface.
- **Adopt a stateless operator API.** Future kernel/operator APIs should take
  `initial_state` and return `final_state` explicitly instead of mutating an
  in-place `state` dict. This is clearer, autograd-friendly, and matches the
  FlashRWKV `rwkv7(..., initial_state=, output_final_state=)` contract. Apply
  this pattern to new ops; migrate existing ones when refactoring.
- **Future: real batch (B>1) support.** Currently there is NO real batching:
  `State` has no batch dim (one `State` = one sequence), `forward` silently
  flattens `[B,T]` to a single `[B*T]` sequence (the benchmark's `BxT` cases
  are just longer single-sequence lengths, B is not real), `decode` handles
  one token, and `CUDAGraph` (like the old `GraphDecoder`) is inherently B=1.
  Real batch needs a dedicated adaptation: `State` gains a batch dim
  (`rnn [B,H,N,N]`, `x [B,C]`), prefill/decode run the batch in parallel, and
  the DPLR recurrence iterates B independent per-sequence states.
  User-approved direction -- DO NOT start this now; revisit when the user
  asks.
