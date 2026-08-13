# Hardware notes

Read this when doing GPU-specific work or benchmarking.

## RTX 3060 (sm_86, 12GB)

Validation/comparison target. The authoritative performance record is
`docs/runs/rtx3060.md`.

## MX450 (sm_75, 2GB)

Old reference GPU (the per-device tuned variant `tl-mx450` was removed on
2026-08-13; details in git history). It is a Turing card with pathological fp16
cuBLAS GEMMs and severe thermal throttling under sustained load (latencies
inflate up to ~50%, p90 >> p10) -- treat single-session relative comparisons
as reliable, absolute numbers as noisy.

- **MX450 (sm_75) has no bf16 tensor cores: do NOT test or benchmark bf16
  here** (tilelang bf16 kernels can fail to compile/lower on this device, e.g.
  "Cannot find var remap for <buffer>" in `StorageLegalizer`). bf16 paths must
  be validated on sm_80+ (RTX 3060 box); MX450 work is fp16-only.
- **Kernels tuned on MX450 must be re-verified on sm_80+.** MX450's
  pathological fp16 cuBLAS makes hand-written serial kernels look good there,
  but sm_86 exposes under-parallelized implementations (see the 9e81fd1
  prefill regression: a serial-over-T oWt GEMV looked fine on MX450 but was
  27x slower than batched `T.gemm` on RTX 3060). Benchmark both cards before
  trusting an MX450-tuned kernel for the 3060 target.
