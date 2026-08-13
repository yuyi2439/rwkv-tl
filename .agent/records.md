# Benchmark and test run records

Read this when updating benchmark or test validation records.

## Benchmark records

Benchmark/test results are recorded in
[docs/runs/rtx3060.md](../docs/runs/rtx3060.md) (the old
`script/benchmark_rwkv7.md` report and `docs/benchmarks/` were removed/merged
into `docs/runs/`). Follow these rules strictly:

- Keep it in Chinese.
- Keep it concise and report-like: benchmark entry script, environment,
  measured results, and short explanations that directly interpret those
  results.
- Do not put exploratory findings, long reasoning, speculative conclusions, or
  operational caveats in the run record.
- Put runtime warnings, environment constraints, and maintenance guidance in
  this records file.
- When a new benchmark/test run is completed, add the numbers to the run
  record and keep the narrative short.

## Test validation records

- Validation is automated: run `pytest test/` against the checkpoint to verify
  correctness. Do not maintain a hand-written per-GPU validation document
  under `docs/` -- test outcomes that matter are captured by the test suite
  itself (the old `docs/validation_*.md` files were removed for this reason).
- Keep correctness gate notes (which model versions / GPUs pass) in AGENTS.md
  or the benchmark report when relevant, not in a dedicated validation doc.
