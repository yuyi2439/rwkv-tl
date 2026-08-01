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

The numbers below were collected on an NVIDIA MX450. They are useful for relative comparisons on the same hardware, but they are not the final target validation numbers.

The next validation run will be on an RTX 3060.

| Case | rwkv_tl | pure_torch | graph_decoder |
|---|---:|---:|---:|
| 1x1 | 58.21 ms / 17.18 tok/s | 22.76 ms / 43.93 tok/s | 8.35 ms / 119.79 tok/s |
| 8x8 | 265.70 ms / 240.87 tok/s | 398.34 ms / 160.67 tok/s | not supported |
| 16x16 | 899.69 ms / 284.54 tok/s | 1341.51 ms / 190.83 tok/s | not supported |

Key points:
- GraphDecoder is best for single-token decode latency.
- rwkv_tl is the only path that supports both batched prefill and decode.
- The pure-torch baseline was substantially improved by batching the prefill path.

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
