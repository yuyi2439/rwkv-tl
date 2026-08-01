# AGENT.md

Working notes for agents working on this repository.

## Goal

Implement and validate faster RWKV7 inference paths in this repo. Keep the implementation correct and verify it with the real benchmark and test scripts.

## Core constraints

- Do not add compatibility shims; edit the implementation directly.
- Verify TileLang and PyTorch APIs before using them.
- Prefer existing project code over new helpers.
- Do not swallow exceptions. Only catch errors when recovery is meaningful.
- Keep new docstrings in src/rwkv_tl short and in English.
- Do not create extra files unless they are clearly necessary.

## Hardware note

This project is scheduled for validation on an RTX 3060 tomorrow. MX450 results are useful for relative comparisons, but they are not the final target numbers. Do not treat MX450-only measurements as the final answer.

## Performance work

- Decode path: fused TMIX/CMIX kernels plus GraphDecoder.
- Prefill path: batched GEMM instead of per-token GEMV where possible.
- Keep correctness first: forward and forward_prefill must use independent state objects in tests.

## Benchmarks

Use the real scripts in script/ rather than ad-hoc snippets.

```bash
cd /home/yuyi2439/rwkv/rwkv-tl
.venv/bin/python -m pytest test/ -v
.venv/bin/python script/benchmark_rwkv7.py --device cuda ...
```

On memory-constrained GPUs, split large sweeps into separate processes. A single process can accumulate too much compile-cache pressure and trigger OOMs.

## Known issues and notes

- The pure-torch baseline was improved by batching prefill work.
- A token-shift aliasing bug existed in the old TMIX path. Any state update that overwrites previous state must happen only after all reads from the old state are complete.
- The benchmark harness should skip per-case OOMs rather than abort the whole sweep.

## Maintenance

Update this file when a new constraint, bug, or performance path is discovered. Keep it concise and actionable.
