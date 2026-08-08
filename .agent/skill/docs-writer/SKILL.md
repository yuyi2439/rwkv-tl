---
name: docs-writer
description: >-
  Rules for maintaining this project's documentation: AGENT.md and
  .agent/skill/ skills. Covers where each finding goes (repo convention vs
  skill vs docs vs skip), what not to write (obvious facts, mistakes logs),
  skill conventions (frontmatter, concise description, scoped content), and
  the key workflow: creating a skill means AGENT.md only gains a reference and
  the related content moves from AGENT.md into the skill.
---

# Project Docs Writer

How to write and maintain this project's `AGENT.md` and `.agent/skill/` files.
Follow these when recording a new finding, convention, or lesson.

## 1. Where each finding goes

Decide placement first; a finding lives in exactly one place.

| Content type | Location |
|---|---|
| TileLang / kernel-writing knowledge (how to use `T.gemm`, `T.dynamic`, `T.macro`, tiling, pitfalls) | skill, e.g. `.agent/skill/tilelang-writer` |
| Repo-specific conventions (project layout, `make_rwkv7` backends, CUDA-graph mechanism, state design, kernel dtype binding) | `AGENT.md` "Project structure and standards" |
| Python-language facts that are project standards (e.g. tilelang DSL files must not use `from __future__ import annotations`) | `AGENT.md` "Project structure and standards" |
| Benchmark results and experiment findings | `docs/` (Chinese, report-like) |
| Facts you can point out without study (e.g. Python `^` is XOR, not power) | nowhere — skip |

Test: if a TileLang kernel-writer would need this to write a kernel, it is a
skill item. If it constrains how *this repo* is organized, it is an AGENT.md
item. If it is general Python/tool trivia anyone would notice, it is nothing.

## 2. What NOT to write

- **Do not document facts you can point out directly without study.** Only
  record findings that required real investigation or measurement (tilelang
  internals, benchmarks, non-obvious behavior). Writing obvious trivia adds
  noise and buries the real findings. Applies to AGENT.md, docs/, and skills.
- **Do not keep a running "mistakes" log.** An already-fixed code bug is not
  worth recording (it will not recur); record only a reusable lesson (a
  workflow rule, an API pitfall, a doc-accuracy check).
- **Suggestions are not requirements.** If something only applies when the user
  runs into it (e.g. a pyright suppression comment), frame it as
  "Suggestion (not a requirement)" inside the skill — do not make it a
  mandatory rule.

## 3. Skill file conventions

- Layout: `.agent/skill/<skill-name>/SKILL.md`; the skill `name` equals the
  directory name and states the purpose (e.g. `tilelang-writer` for writing
  kernels, `agent-skill-writer` for writing guides).
- Frontmatter:
  - `name`: same as the directory.
  - `description`: 1-1024 chars, concise, states the purpose. If the skill
    targets a specific tool version, mark it in the description
    (e.g. "tilelang 0.1.13" — this project's version).
- Body: step-by-step instructions, reusable code snippets, best practices, and
  common pitfalls with the **exact error messages** (so agents can grep-match
  a failure to a fix). English only.
- **Scope the content to the subject.** A tilelang skill contains only tilelang
  knowledge — no general Python trivia (that belongs nowhere, or in AGENT.md as
  a project standard if it is one).
- Arch-specific facts are fine and often clearer than generic names: e.g.
  "sm_86 uses `mma.sync.m16n8k16`" is better than "Ampere uses ...". Do NOT
  add "verified on <hardware>" statements to the body unless they add clarity.

## 4. AGENT.md maintenance

- `AGENT.md` has a `## Skills` section instructing agents to browse
  `.agent/skill/` and read the skill whose `description` matches the task —
  not to re-derive or re-document what the skill already covers. New
  hard-won TileLang findings go into the skill, not AGENT.md.
- **Creating a skill: AGENT.md only gains a reference; content moves INTO the
  skill.** When a topic grows enough to warrant its own skill, (1) create
  `.agent/skill/<name>/SKILL.md`, (2) add a one-line reference to it in
  AGENT.md (e.g. the `## Skills` section or the relevant bullet), and
  (3) **move** the related content out of AGENT.md into the skill — delete it
  from AGENT.md, do not leave a duplicate. AGENT.md keeps only repo-specific
  conventions and pointers; the skill owns the how-to knowledge.
- Repo standards and decisions that affect future work are recorded in the
  same session ("I'll remember" is not an acceptable substitute).
- Keep AGENT.md concise and actionable: no personal notes, speculation, or
  long retrospective writing. English, unless a specific repo report is
  intentionally written in Chinese.

## 5. Workflow

1. On a new finding/convention, place it per section 1 (one place only).
2. If it is a skill item, add it to the matching skill's SKILL.md; if an
   AGENT.md item, add to the relevant section.
3. When creating a NEW skill: create the SKILL.md, add a reference in AGENT.md,
   and MOVE the now-skill-covered content from AGENT.md into the skill (no
   duplication).
4. Follow AGENT.md's git rules: never `git add`/`commit`/`push` without
   explicit user approval; state-changing git operations are always scoped to
   the single approved action.
