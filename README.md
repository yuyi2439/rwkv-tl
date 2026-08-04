# rwkv-tl

RWKV7 inference with TileLang fused CUDA kernels and CUDA Graph. The goal is to make decode and prefill faster than the pure PyTorch baseline and approach the performance of the Albatross reference implementation.

- Model: RWKV7-g1d (0.1B and 0.4B variants)
- Precision: float16 compute with float32 accumulation (DPLR state stays fp32), matching Albatross
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
target validation GPU, after the stateless refactor with the single-shot
`fused_dplr_T` prefill kernel. `rwkv_tl` and `pure_torch` run the eager path;
the benchmark harness routes through the raw methods so a sweep does not
recompile a fresh graph per token count.

| Case | rwkv_tl | pure_torch | graph_decoder |
|---|---:|---:|---:|
| 1x1 | 9.58 ms / 104.41 tok/s | 14.50 ms / 68.98 tok/s | 1.66 ms / 602.63 tok/s |
| 1x32 | 17.90 ms / 1787.72 tok/s | 121.82 ms / 262.68 tok/s | 51.57 ms / 620.50 tok/s |
| 8x8 | 16.09 ms / 3978.86 tok/s | 214.81 ms / 297.94 tok/s | not supported |
| 16x16 | 15.87 ms / 16135.64 tok/s | 969.97 ms / 263.93 tok/s | not supported |

Key points:
- GraphDecoder is best for single-token decode latency.
- rwkv_tl is the only path that supports both batched prefill and decode.
- The single-shot `fused_dplr_T` prefill kernel made prefill latency
  flat across T (0.1B ~15-18 ms for all prefill cases); it now beats
  `pure_torch` by ~28x at 1x128.
- The Albatross reference (faster3a_2607) still leads prefill by ~2.2x
  (7.3 ms on 0.1B 1x128 vs 15.8 ms for rwkv_tl).
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
| 1 | 14.9ms (noisy) | **8.2ms (stable)** |
| 2 | **8.7ms** | 24.0ms |
| 4 | **9.5ms** | 25.0ms |
| 8 | **22.9ms** | 25.2ms |
| 16 | 33.7ms | **28.0ms** |
| 32 | 43.8ms | **32.5ms** |

- **T=1 decode beats faster3a**: CUDA Graph removes launch overhead, giving a
  stable 8.2ms vs faster3a's fluctuating 8.3-23.2ms.
- **T>=16 prefill beats faster3a**; faster3a still leads the tiny-prefill
  regime (T=2/4/8).

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
