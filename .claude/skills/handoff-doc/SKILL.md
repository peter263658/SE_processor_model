---
name: handoff-doc
description: Create a HANDOFF / RESULTS / STATUS / SPEC / FUTURE_WORK markdown doc using this repo's consistent template (TYPE_NAME_YYYY-MM-DD.md, TL;DR table, code-review table, smoke checks, sectioned narrative). Use when the user asks to "write a handoff", "write a results doc", "draft a status report", "write a spec for X", etc.
---

# handoff-doc

This repo has a recurring documentation pattern. Each phase of work produces one or more markdown files following a fixed naming and section convention. Match it.

## Filename convention

`<TYPE>_<NAME>_<YYYY-MM-DD>.md` — uppercase type, snake-uppercase name, ISO date in the title, dropped in the **current variant directory** (e.g., `14_low_power_arch_qat/`), not the repo root.

Valid `<TYPE>` values:

| TYPE | Purpose |
|---|---|
| `HANDOFF` | Pass context to next session/person. State at this moment, what's been tried, what to do next. |
| `RESULTS` | Experiment outcome. Metrics table, verdict, recommendation. Append `_FINAL` when superseding an earlier scaffold. |
| `STATUS` | Live snapshot of an ongoing multi-phase effort. Chronological; preserve in-flight notes. |
| `SPEC` | Design spec for a new algorithm / RTL block. Inputs, outputs, invariants, validation plan. |
| `FUTURE_WORK` | Things deferred from this round with enough context for someone to pick them up. |
| `TODO` | Short-lived task list (consider whether STATUS isn't better). |
| `RTL_HANDOFF` | Specifically for RTL team consumption — Python ref + bit-exact behavior + waveform/edge cases. |
| `ARCH_BENEFIT_COMPARISON` | Comparative analysis across variant dirs (e.g., 09 → 14). |

The date is the date the doc was first written; if you supersede an earlier doc, create a new one with today's date and add `_FINAL` or `_v2`, do not rewrite history. Link from the new doc to the superseded one.

## Section conventions

All types share a top matter:

```markdown
# <TYPE>: <Human title> [— optional suffix like "Final Sign-off"]

Date: <YYYY-MM-DD>[ (final, supersedes [old_doc.md](old_doc.md))]
Base candidate: <ckpt path or commit hash>
Handoff: [<prior handoff>.md](prior_handoff.md)   (omit if not part of a chain)
```

Type-specific bodies (use as a starting structure; cut sections that don't apply):

### HANDOFF
1. `## 1. Code review` — diff table (File / Lines changed / Summary), then blocker/observation list
2. `## 2. Smoke checks` — `py_compile` results, `bash -n` syntax checks, any micro-benchmark
3. `## 3. Current state` — what's in the active ckpt, what flags are set, what gate it last passed
4. `## 4. Open questions for next session`
5. `## 5. Reproduction commands` — exact CLI to resume work

### RESULTS
1. `## 0. TL;DR` — verdict table with columns: Variant / RTL effect / Mode / Δ on each test set / Verdict. Star the recommendation row (`⭐⭐`).
2. `## 1. Setup` — base ckpt, eval sets (VCTK + TIMIT), gate values (full 0.835 / 0.787 etc.)
3. `## 2. Results` — per-variant detail; one subsection per variant
4. `## 3. Cross-cuts` — patterns that span variants (e.g., "QAT helped X but not Y")
5. `## 4. Recommendation` — one paragraph + bullet rationale
6. `## 5. Reproduction`

### STATUS
- Preserve chronological narrative; append rather than overwrite
- Section per phase (`## 1. Phase A`, `## 2. Phase B`, ...) plus a final `## N. Production stack identified` summary
- Note any patches applied mid-run (env vars added, OOM workarounds) so future-self knows why params look the way they do

### SPEC
1. `## 1. Motivation` — what's it replacing, why
2. `## 2. Inputs / Outputs / Invariants` — signature + tolerance + bit-exactness requirement
3. `## 3. Algorithm` — pseudocode or reference Python
4. `## 4. Validation plan` — which `verify_*.py` checks, which MBSTOI gate
5. `## 5. Failure modes`

### RTL_HANDOFF
- Lead with the bit-exact Python reference (literal code block)
- Then waveform diagram for the matmul/datapath
- Then edge cases: saturation, denormal handling, sign behavior
- Always include packet size / group size / scale-application order — these are the things RTL gets wrong

## Tone and content rules

- **Numbers must be exact**: copy MBSTOI / PESQ / Δ values from the actual eval log, do not round or paraphrase. Prefer 4-decimal precision (`0.8354`, not `~0.84`).
- **Gates must be labeled**: every pass/fail verdict must name which gate (20-pair sanity, full VCTK 0.835, full TIMIT 0.787) — see existing docs for why this matters.
- **Link, don't restate**: if a prior doc explained the L-Mul algorithm, link to it instead of re-explaining.
- **No emojis** except the `⭐⭐` recommendation marker in TL;DR tables (this repo's existing convention).
- **Chinese is fine** when the user writes in Chinese; mixed CJK/English is the established style in CLAUDE.md and existing docs. Match what the user uses in the request.
- **No "and finally" / "in conclusion" prose** — the existing docs are dense fact tables, not essays.

## What to do when invoked

1. Ask the user (only if not already clear from context):
   - Which TYPE? (HANDOFF / RESULTS / STATUS / SPEC / FUTURE_WORK / RTL_HANDOFF / ARCH_BENEFIT_COMPARISON)
   - Short NAME slug? (e.g., `P4_LMUL_FEASIBILITY`, `DECODER_GROUPED_P2`)
   - Which variant directory? (default: current working dir if it matches `^\d+_`)
2. Build the filename: `<TYPE>_<NAME>_<YYYY-MM-DD>.md` using today's date.
3. Pre-populate the type-specific skeleton above. Leave clearly-marked `TODO:` placeholders only where you genuinely need data the user hasn't given you.
4. Fill in what you can from conversation context (ckpt paths, recent eval numbers, last commit).
5. Before writing, sanity-check: does a doc with this name already exist? If yes, ask whether to append `_FINAL` / `_v2` / overwrite.
