# rwkv-tl

RWKV7 inference with TileLang fused CUDA kernels and CUDA Graph. The goal is to make decode and prefill faster than the pure PyTorch baseline and approach the performance of the Albatross reference implementation.

- Model: RWKV7-g1d (0.1B and 0.4B variants)
- Precision: bfloat16 compute with float32 accumulation in the DPLR state update
- Paths:
  - Decode (T=1): fused TMIX/CMIX kernels plus GraphDecoder
  - Prefill (T>1): batched TMIX/CMIX kernels that turn token-wise GEMV into batched GEMM

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
target validation GPU. `rwkv_tl` and `pure_torch` run the eager path; the
benchmark harness routes through the raw methods so a sweep does not recompile
a fresh graph per token count.

| Case | rwkv_tl | pure_torch | graph_decoder |
|---|---:|---:|---:|
| 1x1 | 40.48 ms / 24.70 tok/s | 13.74 ms / 72.77 tok/s | 2.11 ms / 473.14 tok/s |
| 1x32 | 118.12 ms / 270.92 tok/s | 98.20 ms / 325.87 tok/s | 58.78 ms / 544.42 tok/s |
| 8x8 | 206.94 ms / 309.26 tok/s | 171.77 ms / 372.58 tok/s | not supported |
| 16x16 | 758.45 ms / 337.53 tok/s | 609.80 ms / 419.81 tok/s | not supported |

Key points:
- GraphDecoder is best for single-token decode latency.
- rwkv_tl is the only path that supports both batched prefill and decode.
- The Albatross reference (faster3a_2607) still leads all cases by a wide
  margin (4.32 ms on 1x1, ~7.5 ms on all prefill cases for 0.1B).
- Compiling `prefill` gives 1.11-1.43x on 0.1B, but recompiles a
  fresh graph per prompt length (minutes), so it stays eager. See
  `script/benchmark_rwkv7.md` and `docs/benchmark_rwkv7_experiments.md`.

## Run benchmark

```bash
.venv/bin/python script/benchmark_rwkv7.py \
  --project-checkpoint <checkpoint.pth> \
  --vocab asset/rwkv_vocab_v20230424.txt \
  --targets rwkv_tl,pure_torch,graph_decoder \
  --device cuda \
  --cases 1x1,1x8,1x32,2x1,8x1,8x8,16x16 \
  --warmup 10 --iters 20
```

On memory-constrained machines, split large sweeps into separate processes to avoid compiler-cache pressure and OOMs.
