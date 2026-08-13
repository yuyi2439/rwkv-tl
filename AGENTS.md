# AGENTS.md

Working notes for agents working on this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It is the canonical,
human-facing project standard (repository layout, kernel-writing reference,
conventions). AGENTS.md only adds agent-specific operating rules on top of it.

This file is the always-read core. Topic files under `.agent/` are read only
when the relevant area is touched -- see the index at the bottom.

## Operating rules

- **Verify before writing — and re-verify after big changes.** Any rule that
  asserts something about the current code (a symbol, a default value, a file
  path, an API split) must be checked against the source before it is written
  -- grep/read the code, then write. Do not state how something "used to" work
  or how it "should" work; stale or invented details mislead readers and are
  worse than omitting the detail. After a large external/pulled refactor,
  check this file's claims line-by-line against the code before trusting the
  commit message. Diff the actual branches, not the narrative.
- Update AGENTS.md when a new constraint, bug, environment limitation, or
  workflow rule is discovered; do not leave important findings only in chat
  history.
- **Proactively record project standards the user states.** When the user
  states a convention, rule, or design preference, record it in the same
  session -- do not leave it only in chat history or ask for confirmation.
- **User-stated facts are authoritative; surface them before claiming
  otherwise.** On 2026-08-13 the user had to remind the agent that large models
  do NOT beat faster3a -- the data was already in `docs/runs/rtx3060.md`, but
  the agent did not surface it. Before claiming performance superiority (or
  implying big-model parity with faster3a), check the performance record
  first; treat user-stated performance constraints as firm.
- **Self-improve skills on user feedback.** When the user raises an issue about
  behavior governed by a skill (`.agent/skill/<name>/SKILL.md`), update that
  skill so it prevents the problem next time.
- **Before cross-linking per-GPU docs, confirm the machines actually match.**
  MX450 (laptop, 2GB) and RTX 3060 (desktop, 12GB) are different machines.
  Verify hardware before asserting a shared test record.
- **Never run git write/state-changing operations on your own** (no `git add`,
  `git commit`, `git reset`, `git restore`, `git rm`, etc.). Reading state via
  git (`git status`, `git diff`, `git log`, `git show`, `git fetch`) is always
  allowed. The ONLY state-changing git operation permitted without approval is
  renaming/moving an already-tracked file (`git mv`). Permission for any other
  git write (commit/push/amend) is ALWAYS temporary and scoped to that single
  action; each commit/push needs its own explicit approval. When in doubt,
  ask.
- **Report incompatibilities; do not fix design choices on your own.** When you
  hit an incompatibility in user-authored code (dtype/API mismatches, a crash
  you can repro), STOP and tell the user directly with a repro, instead of
  silently changing their design. Fixing genuine bugs (undefined behavior,
  crashes) is fine, but prefer flagging + suggesting the one-line fix and let
  the user decide.
- **Ask before design decisions.** Before proposing/implementing an
  architecture change (new params, new weight-storage schemes, refactors
  touching `weight.py` layout), present the plan and ask the user to confirm
  -- they have strong opinions about naming and where logic lives. Confirm
  scope + naming before writing code.

## User preferences and project standards

- Docs and reports under `docs/` are written in Chinese. Source code
  comments/docstrings stay in English and short.
- Correctness validation is automated via `pytest test/`; record test outcomes
  only when they change a decision or are a notable gate, not as a routine log.
- **A refactor (module rename/move, path changes) must update every affected
  reference in code docstrings and project docs (AGENTS.md, CONTRIBUTING.md,
  and everything under `docs/`, including `docs/runs/`).** Fix it in the same
  pass as the rename, and grep for stale names after moving code.
- Checkpoints are located via the `RWKV_CHECKPOINT_PATH` env var /
  `--project-checkpoint` flag; the directory is machine-specific. Tested
  checkpoints: rwkv7-g1d-0.1b, rwkv7-g1d-0.4b. Test the originally-used model
  first, then the others; watch out for OOM.
- **Vocab is in-package data (user-stated 2026-08-13).** The single source is
  `src/rwkv_tl/asset/rwkv_vocab_v20230424.txt`; package code must only
  reference in-package resources (no repo-root/absolute-path fallbacks in
  `src/rwkv_tl`).
- **examples/ conventions (user-stated 2026-08-13).** Example scripts are
  minimal and plain-Python, written in English; `examples/` has no README.
- **Ruff is mandatory (user-stated).** `ruff check` and `ruff format` must
  stay green across `src/`, `test/`, `script/`, and `examples/`.
- **Core is dependency-free; text lives outside core (user-stated
  2026-08-13).** `src/rwkv_tl/core/` holds only the low-level/inference
  modules (`model` / `state` / `tokenizer` / `weight` / `cuda_graph`);
  `CUDAGraph` wraps any `RWKV7Model` (inference layer only). Code in core
  MUST NOT reference anything outside core, while external modules may import
  from core. The upper-layer text wrapper is `rwkv_tl.text_model.
  RWKV7TextModel` (outside core): it COMPOSES a token-level `RWKV7Model` as
  `self.model` (no inheritance) plus a decoupled tokenizer, exposes the full
  inference surface by delegation (`decode` / `prefill` / `forward` / `w`),
  and adds `tokenize` / `detokenize` / `generate(str, decoding params,
  stop=...) -> str` / `chat(messages, ...) -> str` (renders
  `asset/rwkv_chat_template_v20260805.jinja`). `generate` takes a string
  prompt, returns only the newly generated text, and stops early when the
  output contains the `stop` string (the match is truncated away).
  `RWKV7TextModel` is the class the library exposes: `rwkv7()` returns one,
  wrapping `RWKV7TL` / `RWKV7Torch` (optionally CUDA-Graph wrapped).
- **Weights are never stored or duplicated above 16 bit/param.** No fp32
  weights, fp32 weight copies, or fp32-input GEMMs as a performance lever
  (2x weight VRAM). fp32 is allowed only for compute internals: fp32
  accumulation inside kernels, fp32 RNN state (`[H,N,N]`, matches Albatross),
  fp32 intermediate math. The planned memory-savings direction is quantization
  (int8/any4 weights, dequant fused into the hand-written GEMV); see TODO #6.
- `prefill` stays eager: torch.compile of prefill recompiles a fresh graph per
  distinct prompt length (minutes, GPU idle) for only 1.11-1.43x steady-state.
  This was validated on RTX 3060 and is a firm decision -- do not re-enable
  without new evidence.
- Long benchmarks must run as background processes writing to a log file, then
  be monitored -- never as a blocking foreground command that looks frozen.
- If a script appears to hang with idle CPU/GPU, investigate before assuming it
  failed: torch.compile or first-call kernel compilation can idle the GPU for
  minutes.
- When the user says "check it yourself" or "you can do more tests",
  investigate and run any additional worthwhile tests autonomously.

## Project structure and standards (core API contract)

- **`src/rwkv_tl/` is the published library and contains the models.** The
  package must be self-contained: no docstring or comment inside `src/rwkv_tl/`
  may reference `script/` or `docs/` (files that do not ship with the package).
  Models, the tokenizer (vocab packaged in the wheel), sampling, state, and
  CUDA-Graph all live IN the package.
- **User-facing API (firm, user-stated).** `rwkv_tl.rwkv7(path, ...)` returns
  a `RWKV7TextModel` (backend auto: tilelang on CUDA, torch elsewhere; inner
  model CUDA-Graph wrapped when `use_graph=True`); `RWKV7TL` / `RWKV7Torch`
  are the explicit token-level classes. Text entry points:
  `model.generate(text, max_new_tokens=..., stop=...)` for text generation,
  `model.chat(messages, ...)` for template-based chat. All interfaces are
  STATELESS: `State` is passed into `decode`/`prefill`; models never own
  runtime state.
- **Model interface.** `rwkv_tl.core.model.RWKV7Model` is the token-only
  stateless ABC (`decode` / `prefill` / `forward`); `rwkv_tl.text_model.
  RWKV7TextModel` layers the tokenizer + `generate` + `chat` on top. Every
  model implements `RWKV7Model`. Application scripts build models via
  `rwkv_tl.rwkv7(...)` / `rwkv_tl.make_rwkv7(...)`; do not hard-code a
  specific model class into an application script.
- **Kernels are weight-bound factories.** Bound factories take `(C, DTYPE,
  ...)` plus the weights at construction and return a `BoundKernel` whose call
  only takes activations/state; both granularities are exported from
  `kernel/__init__.py`; legacy per-op kernels live in `kernel/old/`. Weight
  binding is at the wrapper level (TileLang has no compile-time tensor
  constants, so kernels still receive weight pointers per launch). Writing
  conventions: `.agent/kernels.md`.
- **Dtype plumbing.** `RWKV7Weight(path, dtype=...)` controls weight precision
  (default `torch.float16`, converts the bf16 checkpoint once at load; pass
  `torch.bfloat16` to keep the raw dtype). `State(..., dtype=...)` must match
  the model dtype. DPLR RNN state is always fp32.
- **Backends.** `rwkv_tl.rwkv7` / `make_rwkv7` accept only `"auto"` (tl on
  CUDA, torch elsewhere), `"tl"`, and `"torch"`. The tuned variants
  (`tl-mx450`/`tl-rtx3060`/`tl-tuned`) and dtype-carrying backend names
  (`"fp16"`/`"bf16"`) were removed on 2026-08-13; weight precision is
  controlled exclusively by `RWKV7Weight(dtype=...)`. `use_graph=True`
  (default) wraps every CUDA model in `CUDAGraph`.
- **`CUDAGraph` is THE CUDA-Graph mechanism.** Wrap any `RWKV7Model` instance:
  `model = CUDAGraph(RWKV7TL(w))`. Build wrapping ONLY by constructing
  `CUDAGraph` directly (no `wrap_model`/`make_graph_cls` helpers); when CUDA
  is available and `use_graph` is not explicitly disabled,
  `rwkv7()`/`make_rwkv7()` wrap the inner model with `CUDAGraph` themselves.
  `RWKV7TextModel.model` is a plain `RWKV7Model` and does not care whether it
  is CUDA-Graph wrapped. Capture is lazy (T=1 decode, per-T prefill up to
  `prefill_graph_max_t`, default 1024); capture requires in-place
  `state["x"]` updates (`copy_`, not rebind); larger T, non-CUDA models, and
  capture failures fall back to eager. The wrapper copies the caller's
  `State` in/out around each replay, so any `State` works and the model stays
  stateless.

## Goal

Implement and validate faster RWKV7 inference paths in this repo. Keep the
implementation correct and verify it with the real benchmark and test scripts.

**Performance scope (user-stated 2026-08-13): beating faster3a on LARGE models
is explicitly NOT a goal.** Measured on RTX 3060: 1.5B loses 1.3-2.1x across
decode and prefill (see `docs/runs/rtx3060.md`); the gap comes from faster3a's
hand-written CUDA kernels (cp.async, row1_exact, split-K) that the tilelang
route cannot close. The project's value is the operator library, quantization
(TODO #6), and ease of use; small models (0.1B/0.4B) are already
competitive/ahead. Do not spend effort chasing big-model inference parity.
Details: `.agent/performance.md`.

**Long-term direction: rwkv-tl must support TRAINING.** All new
operators/optimizations must keep autograd compatibility in mind; prefer
per-op custom ops; CUDA Graph is inference-only. Details:
`.agent/architecture.md`.

## Core constraints

- Do not add compatibility shims; edit the implementation directly.
- Verify TileLang and PyTorch APIs before using them.
- Prefer existing project code over new helpers.
- Do not swallow exceptions. Only catch errors when recovery is meaningful.
- Do not create extra files unless they are clearly necessary.

## Skills

The only portable skill is `tilelang-writer` (`.agent/skill/tilelang-writer/
SKILL.md`): read it before writing/editing a TileLang kernel, and add new
hard-won TileLang findings there, not to AGENTS.md. Repo-specific guidance
lives in `.agent/<topic>.md` files; before updating AGENTS.md, a topic file,
or the skill, read `.agent/docs.md` (placement rules, what not to write).

## Topic files (read when relevant)

| File | Read when |
|---|---|
| [.agent/kernels.md](.agent/kernels.md) | writing/modifying TileLang kernels (GEMV contract, DSL rules, docstring conventions) |
| [.agent/hardware.md](.agent/hardware.md) | GPU-specific work (MX450/3060 differences, bf16 restrictions) |
| [.agent/performance.md](.agent/performance.md) | performance work, benchmarks, or performance claims |
| [.agent/records.md](.agent/records.md) | updating benchmark or test run records |
| [.agent/known-issues.md](.agent/known-issues.md) | touching code with known bugs/regressions, or before re-enabling a skipped test |
| [.agent/architecture.md](.agent/architecture.md) | planning future directions (training, real batch, stateless ops) |
| [.agent/docs.md](.agent/docs.md) | updating AGENTS.md, topic files, the tilelang-writer skill, or docs/ records |

## Reference

[docs](/docs/)
