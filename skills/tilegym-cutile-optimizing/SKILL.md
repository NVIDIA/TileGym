---
name: tilegym-cutile-optimizing
description: >
  Question-gated router for the curated cuTile optimization wiki: kernel-family
  playbooks, measured techniques with caveats, performance patterns, and practical
  language knowledge for Blackwell and Hopper. Use when an agent has inspected the
  current kernel and has a concrete question that could change its plan, when
  measurements point to a specific bottleneck, or when a cuTile language or API
  detail is blocking implementation. Do not use as mandatory onboarding, as the
  source of the first design, or as a fixed checklist of optimizations.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "TileGym Team <TileGym@nvidia.com>"
  tags:
    - cutile
    - performance
    - optimization
    - knowledge-wiki
---

# cuTile optimization skill, wiki assisted

## For human users: how to request kernel optimization

This skill is a reference the agent consults when it has a concrete question —
not an optimization procedure to invoke by name. Do not prompt
"use the tilegym-cutile-optimizing skill to optimize this kernel"; forcing
the skill to load up front defeats its gate and measurably hurts results on
kernels the wiki covers thinly. Instead, state only the goal — preferably via
`/goal`, for example:

```text
/goal Optimize the cuTile implementation of <op> in this repo.
Correctness must keep passing: pytest tests/ops/test_<op>.py --backend cutile
Performance is measured by: python tests/benchmark/bench_<op>.py
Work in a scratch space and leave the original implementation untouched.
Deliver your best correct version as a ready-to-apply patch or file copy in
the scratch space, with measured before/after results, for human review.
```

The agent loads this skill mid-session on its own when a plan-changing
question forms.

The wiki is the repository's canonical cuTile knowledge wiki at
`wiki/cutile-knowledge-wiki/`; all wiki paths below are relative to the
repository root.

The wiki contains prior experience from other kernels, machines, toolchains, and
sessions. Use it to answer a specific question. Do not browse it just because it
is available.

## Core rule

**Think first. Query second. Measure everything.**

Before reading an optimization page:

1. Inspect the current kernel, baseline results, profile data, and your own
   experiment notes.
2. Form your own diagnosis and next-step plan.
3. State one concrete question whose answer could change that plan.
4. Check whether your earlier experiments or measurements already answer it.
5. Open the wiki only if the question remains unresolved.

Use this short checkpoint:

```text
Current diagnosis:
Planned next move:
Unresolved question:
What answer would change the plan:
```

If you cannot fill in `Unresolved question`, continue independent analysis instead
of reading the wiki.

Language and API correctness questions are exempt from this gate: check exact
semantics whenever uncertainty would otherwise produce invalid code.

## Route 0 — language or API question

Use this route for questions about:

* cuTile execution and launch behavior
* tile-space indexing
* `ct.load` versus `ct.gather`
* padding and masking
* numerical behavior
* traced Python restrictions
* compiler hints
* wrapper, specialization, and caching behavior
* exact API availability

Use the official cuTile Python documentation for the formal API reference:
<https://docs.nvidia.com/cuda/cutile-python>. The source and its examples are
at <https://github.com/NVIDIA/cutile-python>.

Use the wiki language page for the practical working model, common conversion
mistakes, and known failure behavior:

```bash
grep -ril "id: cutile-language" wiki/cutile-knowledge-wiki/
```

Ask a narrow question. For example:

```text
Does this runtime page index require gather for the whole tensor, or can I gather
the scalar page id and then use ct.load?
```

Do not use optimization technique pages to answer basic language legality questions.

## Route 1 — kernel shape to kernel-family page

Use this route only after forming your own opening design or obtaining a working
candidate.

Use it to answer questions such as:

* What usually dominates performance for this kernel family?
* Which implementation choices may be missing from my current design?
* Is there a known alternative architecture worth testing?

Read only the relevant parts of:

* `What dominates performance`
* `Applicable techniques`

Do not copy the page as the opening design. Compare it against the design you
formed independently.

| Looks like                              | Page                                           |
| --------------------------------------- | ---------------------------------------------- |
| dense matmul / bmm / grouped GEMM       | `wiki/cutile-knowledge-wiki/kernels/kernel-gemm.md`                  |
| attention prefill (FMHA forward)        | `wiki/cutile-knowledge-wiki/kernels/kernel-attention-prefill.md`     |
| attention decode (few queries, long KV) | `wiki/cutile-knowledge-wiki/kernels/kernel-attention-decode.md`      |
| layer norm / RMS norm / per-slice stats | `wiki/cutile-knowledge-wiki/kernels/kernel-norms.md`                 |
| row softmax, forward or backward        | `wiki/cutile-knowledge-wiki/kernels/kernel-softmax.md`               |
| RoPE / paired 2D rotations              | `wiki/cutile-knowledge-wiki/kernels/kernel-rope.md`                  |
| prefix sum / cumsum / histogram         | `wiki/cutile-knowledge-wiki/kernels/kernel-scan-histogram.md`        |
| embedding / index-select / paged gather | `wiki/cutile-knowledge-wiki/kernels/kernel-gather-scatter.md`        |
| MoE token-routing auxiliaries           | `wiki/cutile-knowledge-wiki/kernels/kernel-moe-align.md`             |

A family match is only a starting point. A recommendation may also depend on the
current architecture, operation role, shape, dtype, GPU, and toolchain.

## Route 2 — measured symptom to pattern page

Use this route only when measurements support a specific symptom.

Available pattern pages include:

* `wiki/cutile-knowledge-wiki/patterns/pattern-memory-bound.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-compute-bound.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-low-sm-utilization.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-register-pressure.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-tail-effect.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-host-overhead-bound.md`
* `wiki/cutile-knowledge-wiki/patterns/pattern-shape-heterogeneity.md`

Before opening a pattern page, state the evidence:

```text
Observed symptom:
Measurement supporting it:
Current suspected cause:
Question for the wiki:
```

Treat the page's candidate techniques as a menu of hypotheses, not a checklist.

Do not try them in page order by default. First check whether the stated cause
actually matches the current code and profile.

For example:

* Do not choose TMA only because the kernel is memory-bound.
* First determine whether the tile is reused, streamed once, block-aligned, or
  genuinely random.
* Do not tune a memory-bound kernel that is already close to peak bandwidth without
  first considering whether fewer bytes must be moved.

A plateau by itself is not a query. Identify a concrete unknown before reading.

## Route 3 — check a specific technique before applying it

Use this route when you already plan to test a specific technique and need to check
its conditions, parameters, or known failure cases.

Technique pages include:

* `wiki/cutile-knowledge-wiki/techniques/tech-tile-size.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-tma-load.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-num-ctas.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-latency-hint.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-occupancy.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-persistent-grid.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-epilogue-fusion.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-group-swizzle.md`
* `wiki/cutile-knowledge-wiki/techniques/tech-ftz-approx.md`

Read only the sections needed to answer the question:

* `What it is`
* `When to use`
* `Caveats`
* the evidence closest to the current case

Before applying the advice, compare:

* the role of the operation, such as mainloop load, epilogue load, or store
* the surrounding kernel architecture
* shape and dtype
* register pressure
* GPU architecture
* cuTile and TileIRAS versions
* the code state in which the historical result was measured

Treat caveats as scoped evidence, not universal bans.

A result measured in one architecture does not prove that the same technique will
have the same effect in another architecture.

## Fallback search

Use fallback search only for an exact term, error message, API, kernel type, or
technique:

```bash
grep -ril "<specific term>" wiki/cutile-knowledge-wiki/
```

The full page inventory is:

```text
wiki/cutile-knowledge-wiki/index.md
```

Do not use directory listings or broad searches as onboarding.

Do not search for generic phrases such as:

```text
how to optimize this kernel
best techniques
what should I try
```

Read one substantive page per question by default. Open a second page only when the
first page creates a concrete follow-up question.

## After reading

Immediately record:

```text
New information:
Why it applies, or may not apply:
Plan before reading:
Plan after reading:
Minimal experiment:
```

If `Plan after reading` is unchanged, stop. Do not keep browsing and do not perform
an experiment merely to comply with the skill.

If the wiki confirms something already present in the plan, treat the read as
confirmation, not as a new optimization direction.

If the page conflicts with a current-board measurement, trust the current
measurement while preserving the exact context of both results.

## API and toolchain check

Before investing in a recipe that uses a specific API or compiler feature:

1. Verify that the current cuTile and TileIRAS versions support it.
2. Check whether the page provides a compatible fallback.
3. If version support is unclear, run the smallest possible compile check first.
4. If the feature requires a newer version, mark the recipe inapplicable to this
   environment and continue with another approach.

A recipe requiring a newer toolchain is not evidence that the current platform is
broken.

Do not let an unavailable preferred path prevent testing a compatible fallback.

## Running the experiment

When wiki information changes the plan:

1. Make one controlled change where possible.
2. Compare against the direct parent candidate.
3. Run correctness checks.
4. Benchmark on the current board.
5. Keep, revise, or revert based on the result.

Do not apply several wiki suggestions at once unless they form one inseparable
design. Otherwise, the result cannot show which suggestion helped.

Work in a scratchpad, not in place: keep experimental kernel variants, one-off
harnesses, and notes in a scratch directory (for example `sandbox/` in the
working tree, or `/tmp`). The original implementation stays untouched end to
end — verify a candidate by temporarily applying it and restoring the original
afterward. Deliver the final best-effort kernel as a ready-to-apply patch or
file copy in the scratch directory, together with measured before/after
results on the official benchmark; whether it replaces the original is a human
decision, not the agent's.

## Workflows

### New kernel

1. Inspect the task and form an independent v1 design.
2. Use Route 0 immediately if language or API semantics are unclear.
3. Before reading optimization pages, write down the independent design.
4. Query Route 1 only with a concrete question that may change that design.
5. Record any plan change caused by the page.
6. Implement and measure.

### Working kernel

1. Inspect the current code and measurements.
2. Identify one likely bottleneck or uncertain decision.
3. Check your earlier measurements and experiment notes.
4. Use Route 1, 2, or 3 for the exact unresolved question.
5. Run one minimal A/B experiment.
6. Record the result with its measured scope (shape, dtype, GPU, code state).

### Plateaued kernel

1. Do not browse the wiki merely because progress stopped.
2. Review failed experiments and current measurements.
3. Identify a specific missing fact, disputed assumption, or untested alternative.
4. Query only that question.
5. If no specific question exists, continue independent exploration.

### Specific optimization already planned

1. State the intended change and expected reason.
2. Use Route 3 to check conditions and caveats.
3. Verify toolchain support.
4. Record whether the page changed the plan.
5. Run the smallest useful experiment.

## Doctrine

Every number, configuration, and recommendation in the wiki is prior evidence from
another kernel, board, toolchain, code state, or day.

* Re-measure on the current system.
* Trust current measurements when they disagree with historical evidence.
* Do not treat a page as an instruction to use a technique.
* Do not treat a caveat as a ban outside its measured scope.
* Do not let the wiki choose your entire search space.
* Profiler-derived, first-principles, and novel approaches are always allowed.
* The absence of a technique from the wiki is not evidence against trying it.
* The wiki is useful only when it supplies information that changes a concrete
  decision.
