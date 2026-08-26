# Project docs maintenance

Read this before updating `AGENTS.md`, `.agent/<topic>.md` files, the
remaining portable skill (`.agent/skill/tilelang-writer`), `README.md` /
`CONTRIBUTING.md`, or `docs/` records. These rules are repo-specific, so they
live as a topic file, not as a portable skill.

## Documentation layers

| Layer | Content | Read |
|---|---|---|
| `AGENTS.md` | always-read operating rules, user preferences, core API contract, topic index | every task |
| `.agent/<topic>.md` | repo-scoped reference details only needed for a specific area (kernels, hardware, performance, records, known issues, architecture, docs) | when touching that area (AGENTS.md "Topic files" index) |
| `.agent/skill/tilelang-writer/SKILL.md` | portable how-to knowledge for writing tilelang kernels in any project | when writing/editing kernels |
| `docs/` | benchmark results and experiment records (Chinese, report-like) | when interpreting results |

Only `tilelang-writer` remains a skill because it is portable tilelang
knowledge. Everything repo-specific belongs in `AGENTS.md` or a topic file.

**Human-facing docs (`README.md`, `CONTRIBUTING.md`, `docs/`) are for people
and must never reference `.agent/` topic files**; any fact they need must be
stated inline or in `docs/`. `.agent/` files are agent-only and are reached
through the AGENTS.md topic index. References are one-directional: topic
files may point at repo paths, never the reverse.

## 1. Where each finding goes

Decide placement first; a finding lives in exactly one place.

| Content type | Location |
|---|---|
| TileLang / kernel-writing knowledge (how to use `T.gemm`, `T.dynamic`, `T.macro`, tiling, pitfalls) | `.agent/skill/tilelang-writer` |
| Repo-specific conventions (project layout, `rwkv7_model`/`rwkv7` backends, CUDA-graph mechanism, state design, kernel dtype binding) | AGENTS.md "Project structure and standards", or `.agent/kernels.md` / `.agent/architecture.md` with an AGENTS.md pointer |
| Repo code/documentation standards — how *this repo* writes and documents its code (docstring `Args:` conventions, where parameter/layout requirements go, internal tuning vars as comments, naming rules) | `.agent/kernels.md` (via AGENTS.md pointer) |
| Requirements about how documentation and skills are written | this file, `.agent/docs.md` |
| Python-language facts that are project standards (e.g. tilelang DSL files must not use `from __future__ import annotations`) | `.agent/kernels.md` (via AGENTS.md pointer) |
| Benchmark results and experiment findings | `docs/` (Chinese, report-like) |
| GPU/hardware-specific constraints and caveats | `.agent/hardware.md` (via AGENTS.md pointer) |
| Benchmark/test record maintenance rules | `.agent/records.md` (via AGENTS.md pointer) |
| Known bugs, regressions, and historical analysis | `.agent/known-issues.md` (via AGENTS.md pointer) |
| Facts you can point out without study (e.g. Python `^` is XOR, not power) | nowhere — skip |

Test: if a TileLang kernel-writer would need this to write a kernel, it is a
skill item. If it constrains how *this repo* is organized, or how its code is
written or documented, it is a repo standard (AGENTS.md core or a topic file).
If it is general Python/tool trivia anyone would notice, it is nothing.

Key disambiguation: **a rule about this repo's own API design is a repo
standard, not a skill item.** Even when it concerns kernel code (e.g. the
`gemv_main_macro` compute / `gemv_macro` store split, or how fused kernels are
organized), if the API/factory is defined by this repo and would not exist in
another project, it goes in AGENTS.md / `.agent/kernels.md`. A skill only
carries knowledge that applies to writing TileLang kernels in *any* project
using tilelang itself — never repo-specific symbols or conventions.

## 2. What NOT to write

- **Do not document facts you can point out directly without study.** Only
  record findings that required real investigation or measurement (tilelang
  internals, benchmarks, non-obvious behavior). Writing obvious trivia adds
  noise and buries the real findings. Applies to AGENTS.md, topic files,
  docs/, and the skill.
- **Do not keep a running "mistakes" log.** An already-fixed code bug is not
  worth recording (it will not recur); record only a reusable lesson (a
  workflow rule, an API pitfall, a doc-accuracy check).
- **Suggestions are not requirements.** If something only applies when the user
  runs into it (e.g. a pyright suppression comment), frame it as
  "Suggestion (not a requirement)" — do not make it a mandatory rule.

## 3. Topic reference files (`.agent/<topic>.md`)

- Repo-scoped reference docs for content that is not needed on every task.
  Existing topics: `kernels`, `hardware`, `performance`, `records`,
  `known-issues`, `architecture`, `docs`.
- AGENTS.md keeps the always-read core plus a "Topic files" index with a
  one-line "read when" hint per file. When a topic is only occasionally
  needed, move the detail into its topic file and leave the pointer in
  AGENTS.md; do not duplicate content.
- Topic files MAY reference repo paths (`docs/`, code, scripts).
- Main content is English; quoted references (e.g. Chinese section titles in
  `docs/QA.md`) are fine.

## 4. Maintaining the tilelang-writer skill

- Layout: `.agent/skill/tilelang-writer/SKILL.md`; the skill `name` equals the
  directory name and states the purpose.
- Frontmatter:
  - `name`: same as the directory.
  - `description`: 1-1024 chars, concise, states the purpose. If the skill
    targets a specific tool version, mark it in the description
    (e.g. "tilelang 0.1.13" — this project's version).
- Body: step-by-step instructions, reusable code snippets, best practices, and
  common pitfalls with the **exact error messages** (so agents can grep-match
  a failure to a fix). English only.
- **The skill is portable and self-contained: it must not reference anything
  outside the skill** — no `AGENTS.md`/`CONTRIBUTING.md`/`docs/` links, no
  pointing at topic files, no "this is an AGENTS.md item" notes. It stands
  alone and must be usable in any project. If a rule is actually a repo
  standard, it belongs in AGENTS.md or a topic file, and the skill stays
  silent about it. When you need to move content out of the skill, delete it
  from the skill completely — do not leave a pointer.
- **Scope the content to the subject.** The tilelang skill contains only
  tilelang knowledge — no general Python trivia (that belongs nowhere, or in
  `.agent/kernels.md` as a project standard if it is one).
- Arch-specific facts are fine and often clearer than generic names: e.g.
  "sm_86 uses `mma.sync.m16n8k16`" is better than "Ampere uses ...". Do NOT
  add "verified on <hardware>" statements to the body unless they add clarity.

## 5. AGENTS.md maintenance

- `AGENTS.md` has a `## Skills` section instructing agents to read
  `tilelang-writer` when the task matches — not to re-derive or re-document
  what the skill already covers. New hard-won TileLang findings go into the
  skill, not AGENTS.md.
- **Creating a topic file: AGENTS.md only gains a reference; content moves
  INTO the topic file.** When a topic is only needed occasionally, (1) create
  `.agent/<topic>.md`, (2) add a one-line pointer + "read when" hint to
  AGENTS.md's "Topic files" index, and (3) **move** the related content out of
  AGENTS.md — delete it from AGENTS.md, do not leave a duplicate.
- Repo standards and decisions that affect future work are recorded in the
  same session ("I'll remember" is not an acceptable substitute).
- Keep AGENTS.md concise and actionable: no personal notes, speculation, or
  long retrospective writing. English main content; quoted references (e.g.
  Chinese section titles) are fine.

## 6. Workflow

1. On a new finding/convention, place it per section 1 (one place only).
2. **Verify code facts before writing them.** Any documented symbol, default
   value, path, or API split must be grep/read-checked against the source
   first; do not write from memory or describe how things "used to" work.
3. If it is a skill item, add it to the tilelang-writer skill; if a repo
   standard, add it to AGENTS.md or the matching topic file.
4. When creating a NEW topic file: create it, add a reference in AGENTS.md,
   and MOVE the now-covered content out of AGENTS.md (no duplication).
5. Follow AGENTS.md's git rules: never `git add`/`commit`/`push` without
   explicit user approval; state-changing git operations are always scoped to
   the single approved action.
