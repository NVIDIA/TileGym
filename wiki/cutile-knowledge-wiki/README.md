<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# cuTile Knowledge Wiki

A **static, results-free** knowledge base for agents optimizing cuTile kernels. Pages describe what
things *are* and *when they apply* — kernel and technique descriptions that hardly age, not experiment
records. Measured findings survive only as scoped one-line evidence bullets under the evidence policy
below.

## How to consume (agents)

1. Start at `index.md` for the page inventory.
2. Before writing or optimizing a kernel: read `languages/cutile-language.md` (mental model + rules
   that bite), then the matching `kernels/` page (computational shape → which techniques apply and why).
3. When a profile or symptom points somewhere but not to a specific fix: read the matching `patterns/` page
   (symptom → likely causes → candidate techniques) to route from the bottleneck to the technique to try.
4. Before trying a candidate optimization: read its `techniques/` page (When to use / Caveats /
   Evidence); the caveats carry the measured boundaries and scoped counter-signals.
5. Evidence bullets on pages are scoped to the arch/board/op they name — treat them as priors to
   re-measure on your board, never as answers to adopt.

## Page contracts (lint-enforced)

Frontmatter (all kinds): `id`, `kind`, `title`, `summary` (one line).

| kind | required sections |
|---|---|
| `language` | Overview · topical sections (execution, tiles, memory, numerics, traced subset, hints/wrapper), each = short model + its rules |
| `kernel` | What it computes · Computational shape · What dominates performance · Applicable techniques · Where it lives (repo pointers) |
| `technique` | What it is · Pattern · When to use · Caveats · Evidence |
| `pattern` | Symptom · Likely causes · Candidate techniques · Caveats |

No `Current state`, `Open ideas`, `Tried`, or `Track record` sections anywhere (lint-enforced) — that
is working state from a specific optimization run, not durable knowledge; it stays out of the corpus.

## Evidence policy

Every evidence bullet carries scope (arch/board/op). Where the evidence is code, it cites a portable
`reference/` snapshot; a measurement with no reproducible anchor wears an explicit "previous
measurement" label. No unscoped numbers; no narrative history.

**Provenance tags (lint-enforced).** Every evidence bullet with measurement content ends with a
`[YYYY-MM, <arch>, <toolkit verbatim>, N=<k>]` tag — the date is mandatory (it is the staleness handle:
an undated claim can never expire), the architecture is expected (missing arch is a lint warning, never
fabricate one), the toolkit string is pasted verbatim if one is available (the point is ordering, not
semantics), and `N=` records how many independent kernels/boards support the claim. Batch-1 bullets
were backfilled with `[2026-07]`, the corpus's landing date.

**Claim basis (lint-enforced).** Technique pages declare `basis:` in frontmatter:
`source-semantics` (traceable to compiler/API source — cannot be wrong, only outdated),
`measured, N=<k> ...` (observed on k independent kernels), or `single-observation` (one kernel, one day —
a hypothesis wearing a claim's clothes; never/always directives on such a page draw a lint warning).
Batch-1 pages predate grading and carry `basis: ungraded-batch-1`.

## Reference snapshots (mediated access)

`reference/` holds frozen code snapshots so the wiki works standalone. They are corpus pages of kind
`reference` (frontmatter id, fetchable via `run.py page <id>`, searched by `run.py grep` so
symbol-level queries land in real code) but they are deliberately **absent from index.md**: agents are
meant to reach them through the technique/kernel pages that cite them, or through a grep hit — not by
browsing. Their `used_by:` frontmatter closes the loop back to the citing pages.

## How knowledge is added (humans)

The wiki is static. New pages enter only via the add-knowledge workflow: draft into `staging/`, a review agent
writes `staging/<id>.review.md`, a human approves with `python3 run.py approve <id>`.

## Tooling

```
python3 run.py lint      # contracts, evidence rules, index freshness — must be 0 errors, 0 warnings
python3 run.py index     # regenerate index.md
python3 run.py new <kind> <id>   # scaffold a staging candidate
python3 run.py approve <id>      # move staged page into the corpus (human step)
```
