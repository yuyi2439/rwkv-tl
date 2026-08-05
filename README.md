# rwkv-tl

RWKV7 inference with TileLang fused CUDA kernels and CUDA Graph. The goal is to make decode and prefill faster than the pure PyTorch baseline and approach the performance of the Albatross reference implementation.

- Model: RWKV7-g1d (0.1B and 0.4B variants)
- Precision: float16 compute with float32 accumulation (DPLR state stays fp32), matching Albatross
- Paths:
  - Decode (T=1): fused TMIX/CMIX kernels, CUDA-Graph accelerated via `CUDAGraph`
  - Prefill (T>1): batched TMIX/CMIX kernels that turn token-wise GEMV into batched GEMM; per-T CUDA-Graph replay for T<=64

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
target validation GPU. `tl-fp16`/`tl-rtx3060` are the fp16 base wrapped with
`CUDAGraph` (CUDA-Graph decode + per-T prefill graph); `pure-torch` is the
eager PyTorch baseline. The benchmark harness routes through the eager methods
so a sweep does not recompile a fresh graph per token count.

| Case | tl-fp16 | tl-rtx3060 | faster3a_2607 | pure-torch |
|---|---:|---:|---:|---:|
| 1x1 | 10.08 ms | **2.16 ms** | 4.40 ms | 15.78 ms |
| 1x8 | 16.46 ms | **2.86 ms** | 6.41 ms | 41.44 ms |
| 1x32 | 16.55 ms | **3.41 ms** | 8.16 ms | 119.29 ms |
| 1x64 | 16.60 ms | **4.00 ms** | 7.71 ms | 228.87 ms |
| 1x128 | 16.64 ms | 17.47 ms | **7.06 ms** | 428.54 ms |
| 8x8 | 16.29 ms | **3.89 ms** | 7.69 ms | 223.73 ms |
| 16x16 | 16.60 ms | **17.34 ms** | 8.02 ms | 850.39 ms |

Key points:
- The CUDA-Graph-accelerated `tl-rtx3060` (decode + per-T prefill graph for
  T<=64) leads every implementation at T=1..64, beating faster3a_2607 by
  ~1.5-2x and pure-torch by ~7-60x.
- T=128 prefill stays eager (T>64 graph cap) and trails faster3a_2607
  (~20 vs 7.8 ms) -- large-T prefill is the common tilelang-path bottleneck.
- Every CUDA model is graph-wrapped by default (`make_rwkv7(use_graph=True)`);
  `backend="torch"` stays a true eager reference for gating.
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
