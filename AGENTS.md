# AGENTS.md

Working notes for agents working on this repository.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It is the canonical,
human-facing project standard (repository layout, kernel-writing reference,
conventions). AGENTS.md only adds agent-specific operating rules on top of it.

This file is the always-read core: operating rules, user preferences, and the
firm API highlights. Detailed descriptions live in topic files under
`.agent/` (read only when the relevant area is touched -- see the index at
the bottom). A fact lives in exactly one place; when a feature changes,
update the topic file that owns the detail and keep AGENTS.md pointers short.

## Important Rule

### KISS and First Principles

Follow the KISS principle and reason from first principles during development. Start by identifying the real problem, required behavior, and smallest useful change before adding code. Do not pile on features, configuration switches, abstractions, dependencies, or compatibility layers unless they directly solve the current problem and have clear evidence of need.

Prefer the simplest implementation that is correct, maintainable, and consistent with the existing codebase. If a broader design seems attractive, reduce it to the essential behavior needed now and leave optional expansion for a later, explicit requirement.

### No Unnecessary Helpers

Prioritize inline implementation over abstraction. Avoid over-engineering and do not create helper functions unless absolutely necessary.

1. **Inline-First Rule**: If a logic block can be implemented directly within the main function without breaking overall readability, **do not** extract it into a new helper function.
2. **Strict Justification for Helpers**: You may only create a separate helper function if it meets at least one of these criteria:
   - **High Reuse**: The exact same logic is repeated across **3 or more** different locations.
   - **Extreme Complexity**: Inlining the logic makes the main function too long (e.g., >50 lines) or severely derails the main execution flow.
3. **No Fragmentation**: Do not split continuous linear logic (e.g., a single API call, simple form validation, or one-time data formatting) into tiny functions just for the sake of "clean code."
4. **Keep Context Compact**: Handle edge cases, error catching, and logging directly inside the main function block instead of offloading them.
5. **Refactoring Constraint**: When modifying existing code, do not alter the current function structure or extract code into new helpers unless the existing code already violates the complexity or reuse rules above.

## Operating rules

- **Verify before writing — and re-verify after big changes.** Any rule that
  asserts something about the current code (a symbol, a default value, a file
  path, an API split) must be checked against the source before it is written
  -- grep/read the code, then write. Do not state how something "used to" work
  or how it "should" work; stale or invented details mislead readers and are
  worse than omitting the detail. After a large external/pulled refactor,
  check this file's claims line-by-line against the code before trusting the
  commit message. Diff the actual branches, not the narrative.
- Record new constraints, bugs, environment limitations, and workflow rules
  in the owning topic file (AGENTS.md only for always-read rules); do not
  leave important findings only in chat history.
- **Proactively record project standards the user states.** When the user
  states a convention, rule, or design preference, record it in the same
  session -- do not leave it only in chat history or ask for confirmation.
- **User-stated facts are authoritative; surface them before claiming
  otherwise.** On 2026-08-13 the user had to remind the agent that large models
  do NOT beat faster3a -- the data was already in the performance record (now
  `.agent/performance.md`), but the agent did not surface it. Before claiming
  performance superiority (or implying big-model parity with faster3a), check
  the performance record first; treat user-stated performance constraints as
  firm.
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
  `.agent/` topic files, and everything under `docs/`).** Fix it in the same
  pass as the rename, and grep for stale names after moving code.
- **Ruff is mandatory (user-stated).** `ruff check` and `ruff format` must
  stay green across `src/`, `test/`, `script/`, and `examples/`.
- **examples/ conventions (user-stated 2026-08-13).** Example scripts are
  minimal and plain-Python, written in English; `examples/` has no README.
- **Vocab is in-package data (user-stated 2026-08-13).** The single source is
  `src/rwkv_tl/asset/rwkv_vocab_v20230424.txt`; package code must only
  reference in-package resources (no repo-root/absolute-path fallbacks).
- **Weights are never stored or duplicated above 16 bit/param.** fp32 is
  allowed only for compute internals; the quantization direction is TODO #6.
  Details: `.agent/architecture.md`.
- **Checkpoints.** Located via `RWKV_CHECKPOINT_PATH` / `--project-checkpoint`
  (machine-specific). Tested models and run rules: `.agent/records.md`.
- **`prefill` stays eager.** torch.compile of prefill recompiles a fresh graph
  per distinct prompt length (minutes, GPU idle) for only 1.11-1.43x; firm
  decision -- do not re-enable without new evidence. Details:
  `.agent/performance.md`.
- Long benchmarks must run as background processes writing to a log file, then
  be monitored -- never as a blocking foreground command that looks frozen.
- If a script appears to hang with idle CPU/GPU, investigate before assuming it
  failed: torch.compile or first-call kernel compilation can idle the GPU for
  minutes.
- When the user says "check it yourself" or "you can do more tests",
  investigate and run any additional worthwhile tests autonomously.

## Project structure and standards (core API contract)

The full contract (entry points, model interface, dtype plumbing, backends,
CUDA-Graph, text-layer composition) lives in `.agent/architecture.md`.
Always-relevant highlights:

- **`src/rwkv_tl/` is the published library.** Models, the tokenizer, sampling,
  state, and CUDA-Graph all live in the package; nothing inside `src/rwkv_tl/`
  may reference `script/` or `docs/` (docstring conventions:
  `.agent/kernels.md`).
- **`core/` is dependency-free** (low-level inference modules only); the text
  wrapper lives outside core and composes the token model. Details:
  `.agent/architecture.md`.
- **User-facing API (firm, user-stated).** Callers create the `RWKV7Weight`
  manually; `rwkv_tl.rwkv7_model(w, backend=...)` maps it to the token-level
  `RWKV7Model`, and `rwkv_tl.rwkv7(w, backend=...)` is the one-call
  `RWKV7TextModel` form. `backend` is REQUIRED (`"tl"` / `"torch"`, no
  auto-detection). All interfaces are STATELESS: `State` is passed into
  `decode`/`prefill`; models never own runtime state. Details:
  `.agent/architecture.md`.
- **Kernels are weight-bound factories.** Bound factories take hyperparameters
  + weights at construction and return a `BoundKernel`; legacy per-op kernels
  live in `kernel/old/`. Details: `.agent/kernels.md`.
- **`CUDAGraph` is THE CUDA-Graph mechanism.** Conditional wrapping goes
  through `try_cuda_graph`; `CUDAGraph` itself is CUDA-only. Details:
  `.agent/architecture.md`.

## Goal

Implement and validate faster RWKV7 inference paths in this repo. Keep the
implementation correct and verify it with the real benchmark and test scripts.

- **Performance scope (user-stated 2026-08-13):** beating faster3a on LARGE
  models is explicitly NOT a goal (1.5B loses 1.3-2.1x; the gap is faster3a's
  hand-written CUDA kernels). Do not spend effort chasing big-model inference
  parity. Details: `.agent/performance.md`.
- **Long-term direction: rwkv-tl must support TRAINING.** New operators must
  keep autograd compatibility in mind; prefer per-op custom ops; CUDA-Graph is
  inference-only. Details: `.agent/architecture.md`.

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

| File                                             | Read when                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [.agent/kernels.md](.agent/kernels.md)           | writing/modifying TileLang kernels (GEMV contract, DSL rules, docstring conventions) |
| [.agent/hardware.md](.agent/hardware.md)         | GPU-specific work (MX450/3060 differences, bf16 restrictions)                        |
| [.agent/performance.md](.agent/performance.md)   | performance work, benchmarks, or performance claims                                  |
| [.agent/records.md](.agent/records.md)           | benchmark/test records, checkpoints, or tested models                                |
| [.agent/known-issues.md](.agent/known-issues.md) | touching code with known bugs/regressions, or before re-enabling a skipped test      |
| [.agent/architecture.md](.agent/architecture.md) | architecture changes, the model API/entry-point surface, or CUDA-Graph design        |
| [.agent/docs.md](.agent/docs.md)                 | updating AGENTS.md, topic files, README/CONTRIBUTING, the tilelang-writer skill, or docs/ records |

## Reference

[docs](/docs/)
