# Benchmark and test run records

Read this when updating benchmark or test validation records.

## Checkpoints

- Checkpoints are located via the `RWKV_CHECKPOINT_PATH` env var /
  `--project-checkpoint` flag; the directory is machine-specific.
- Tested checkpoints: rwkv7-g1d-0.1b, rwkv7-g1d-0.4b. Test the
  originally-used model first, then the others; watch out for OOM.

## Benchmark records

Per-run benchmark tables are no longer kept under `docs/` (the RTX 3060 run
record was removed on 2026-08-20; user-stated: old test data no longer
needed). Durable conclusions and performance guidance live in
[performance.md](performance.md). Follow these rules strictly:

- Keep conclusions concise, in English, and actionable: what was measured,
  why, and what changed.
- Do not preserve raw benchmark tables or historical per-case numbers.
- Do not put exploratory findings, long reasoning, or speculative conclusions
  in the conclusions file.
- Put runtime warnings, environment constraints, and maintenance guidance in
  this records file.

## Test validation records

- Validation is automated: run `pytest test/` against the checkpoint to verify
  correctness. Do not maintain a hand-written per-GPU validation document
  under `docs/` -- test outcomes that matter are captured by the test suite
  itself (the old `docs/validation_*.md` files were removed for this reason).
- Keep correctness gate notes (which model versions / GPUs pass) in AGENTS.md
  or the benchmark report when relevant, not in a dedicated validation doc.
