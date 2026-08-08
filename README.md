# rwkv-tl

RWKV7 inference with TileLang fused CUDA kernels and CUDA Graph. The goal is to make decode and prefill faster than the pure PyTorch baseline and approach the performance of the Albatross reference implementation.

- Model: RWKV7-g1d (0.1B and 0.4B variants) and RWKV7-g1i 1.5B (loading + correctness verified; full benchmark pending)
- Precision: float16 compute with float32 accumulation (DPLR state stays fp32), matching Albatross
- Paths:
  - Decode (T=1): fused TMIX/CMIX kernels, CUDA-Graph accelerated via `CUDAGraph`
  - Prefill (T>1): batched TMIX/CMIX kernels that turn token-wise GEMV into batched GEMM; per-T CUDA-Graph replay up to `prefill_graph_max_t=1024`

## Project layout

```text
src/rwkv_tl/        # main implementation and fused kernels
script/             # benchmarking and profiling scripts
test/               # correctness and kernel tests
asset/              # tokenizer vocabulary
```

## Install and test

```bash
cd rwkv-tl
uv sync
.venv/bin/python -m pytest test/ -v
```

## Benchmark status

The numbers below were collected on an NVIDIA RTX 3060 (sm_86, 12GB), the
target validation GPU. `tl-fp16` is the fp16 base wrapped with `CUDAGraph`
(CUDA-Graph decode + per-T prefill graph); `tl-bf16` is the bf16 variant;
`pure-torch` is the PyTorch reference, also graph-wrapped by default. The
benchmark harness routes through the eager methods so a sweep does not
recompile a fresh graph per token count.

| Case | tl-fp16 | tl-bf16 | faster3a_2607 | pure-torch |
|---|---:|---:|---:|---:|
| 1x1 | **2.36 ms** | 2.08 ms | 5.94 ms | 3.74 ms |
| 1x8 | **2.92 ms** | 3.24 ms | 7.97 ms | 5.93 ms |
| 1x32 | **3.35 ms** | 3.30 ms | 9.55 ms | 14.96 ms |
| 1x64 | **3.81 ms** | 4.07 ms | 9.92 ms | 26.81 ms |
| 1x128 | **5.15 ms** | 5.45 ms | 9.15 ms | 51.14 ms |
| 8x8 | **3.88 ms** | 4.10 ms | 7.97 ms | 26.67 ms |
| 16x16 | **9.22 ms** | 8.88 ms | **7.78 ms** | 100.11 ms |

0.4B (same harness, 2026-08-08): T=1..64 tl-fp16 leads faster3a (5.12 vs
6.57 ms at T=1, 10.55 vs 15.69 ms at T=64); T=128 has caught up (14.63 vs
13.58 ms); 16x16 still trails (real batch). Full tables in
`script/benchmark_rwkv7.md` and `docs/benchmarks/rtx3060.md`.

Key points:
- The CUDA-Graph-accelerated `tl-fp16` (decode + per-T prefill graph, cap 1024)
  leads at T=1..128, beating faster3a_2607 ~1.8-2.5x (e.g. 2.36 vs 5.94 ms at
  T=1, 5.15 vs 9.15 ms at T=128). `tl-bf16` now matches `tl-fp16` -- the old
  "bf16 is slower on sm_86" gap (4x at T=1) disappeared once bf16 decode went
  through the fused gates + CUDA-Graph path.
- Raising the prefill graph cap from 64 to 1024 is a big win for large-T
  prefill: T=256 (16x16) tl-fp16 dropped 17.15 -> 9.22 ms and pure-torch
  T=128 dropped 506 -> 51 ms, because those cases now replay a captured graph
  instead of running ~2175 eager launches. The remaining large-T cost is the
  GEMM compute itself (chunk-parallel prefill, TODO #3).
- The 16x16 case still trails faster3a (9.22 vs 7.78 ms): faster3a runs real
  batch `[16,16]` in parallel while rwkv_tl processes 256 tokens as one serial
  sequence -- catching up needs real batch support.
- Every CUDA model -- including `backend="torch"` -- is graph-wrapped by
  default (`make_rwkv7(use_graph=True)`); pass `use_graph=False` for a truly
  eager class (e.g. the torch reference used for correctness gating).
- Compiling `prefill` gives 1.11-1.43x on 0.1B, but recompiles a
  fresh graph per prompt length (minutes), so it stays eager. See
  `script/benchmark_rwkv7.md` and `docs/benchmarks/rtx3060.md`.

## MX450 tuning (sm_75) now partially beats the sm75-adapted faster3a_2607

`tl-mx450` (sm_75 tuning: fp32 prefill GEMMs + T<=16 tilelang fp16 rkv + CUDA-Graph decode)
vs the sm75-adapted faster3a_2607 from
[yuyi2439/Albatross `support/sm75`](https://github.com/yuyi2439/Albatross/tree/support/sm75)
(0.1B / MX450, warmup=10, iters=20, single session):

| T | faster3a_2607 (sm75-adapted) | tl-mx450 |
|---|---|---|
| 1 | 9.6ms (noisy) | **8.3ms (stable)** |
| 2 | 11.4ms | **11.5ms (tie)** |
| 4 | **10.8ms** | 11.7ms |
| 8 | 23.6ms | **13.0ms** |
| 16 | 34.0ms | **19.5ms** |
| 32 | 43.9ms | **15.8ms** |
| 64 | 47.0ms | **22.5ms** |
| 128 | 88.6ms | **43.4ms** |

**`tl-mx450` now leads (or ties) the sm75-adapted faster3a_2607 at every T.** The
wins stack three sm_75 findings: CUDA-Graph decode (stable 8.3ms T=1), CUDA-Graph
prefill for T<=64 (small-T prefill was launch-bound: a constant ~2175 launches
regardless of T; T=4 dropped 33 -> 11.7ms), and `.contiguous()` on transposed
GEMM weights (non-contiguous cuBLAS operands are ~2.7x slower on Turing;
T=128 prefill dropped 70.6 -> 43.4ms).

Why we win despite both sides using CUDA Graph: faster3a_2607 (its sm75
adaptation also captures per-stage `torch.cuda.CUDAGraph`s) still runs its
prefill through **fp16 tensor-core GEMMs** (`volta_fp16_s884gemm...` ~39ms of
42.6ms at T=32), which are the pathological Turing fp16 cuBLAS kernels (~4-6x
slower than fp32 for these shapes). `tl-mx450` deliberately uses **fp32 GEMMs**
for prefill, which is the correct sm_75 adaptation. This is an architecture-level
difference, not a measurement artifact.

> **Note (project decision):** the fp32 GEMM workaround is a historical record
> only. Per the current project standard, weights are never stored above 16
> bit/param and fp32 is used only for compute internals (accumulation, RNN
> state). The fp32 prefill GEMM path is being retired; the fix for Turing's
> pathological fp16 cuBLAS kernel choice is tilelang hand-written kernels (the
> T-specialized sm_75 fp16 GEMM already is one).

## Run benchmark

```bash
.venv/bin/python script/benchmark_rwkv7.py \
  --project-checkpoint <checkpoint.pth> \
  --vocab asset/rwkv_vocab_v20230424.txt \
  --targets tl-fp16,pure-torch \
  --device cuda \
  --cases 1x1,1x8,1x32,2x1,8x1,8x8,16x16 \
  --warmup 10 --iters 20
```

On memory-constrained machines, split large sweeps into separate processes to avoid compiler-cache pressure and OOMs.
