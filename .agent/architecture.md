# Architecture direction

Read this when planning architecture changes or new features.

## Long-term: support TRAINING

All new operators/optimizations must keep autograd compatibility in mind:
- The fused kernels (`cmix_decode`, `tmix_decode`, ...) are plain tilelang
  kernels called from `rwkv_tl.rwkv7_tl`; training support will need explicit
  backward definitions (a future `torch.library` registration path), not
  autograd through the raw kernel calls.
- CUDA Graph (`CUDAGraph` in `rwkv_tl/cuda_graph.py`) is INFERENCE-ONLY by
  design: it captures the forward launch sequence and does not rebuild an
  autograd graph (replay does not record gradients, fixed buffers conflict
  with autograd's dynamic graph). Do not route anything training-relevant
  through it. `make_rwkv7(..., use_graph=True)` (default) integrates it as the
  `decode`/`prefill` path via a stateless copy-in/out around a fixed shadow
  state.
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
  graph-wrapped via `make_rwkv7(use_graph=True)`) on the RTX 3060 before
  settling the default. Requires the RTX 3060 box (not the MX450 laptop).

- **DONE — models live in the library (0.2 refactor).** `rwkv_tl.rwkv7_tl.RWKV7TL`
  (fused tilelang kernels, fp16/bf16) and `rwkv_tl.rwkv7_torch.RWKV7Torch`
  (pure torch reference, CPU-capable) are part of the published package; the
  pure-torch functional ops (`time_mix` / `channel_mix` / `time_mix_batch` /
  `channel_mix_batch`) live in `rwkv_tl.rwkv7_torch`. `RWKV7TL` binds weights
  to `BoundKernel` wrappers at construction and compiles kernels lazily.
  `rwkv_tl.cuda_graph.CUDAGraph` wraps any `RWKV7Model` instance and provides
  CUDA-Graph decode + per-T prefill. `rwkv_tl.rwkv7(path, ...)` /
  `make_rwkv7(device, backend=...)` build models; all tilelang backends
  resolve to `RWKV7TL` (the per-device `tuned` variants were deleted with the
  fp32-GEMM workaround). The vocab file is packaged in the wheel
  (`rwkv_tl/asset/rwkv_vocab_v20230424.txt`); `Tokenizer()` uses it by
  default.
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
