# Evaluation Report

Evaluation of the `tilegym-converting-cutile-to-triton` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `tilegym-converting-cutile-to-triton`
- Evaluation date: 2026-05-29
- NVSkills-Eval profile: `external`
- Overall verdict: FAIL
- Tier 3 live agent evaluation: not available in this report

## Agents Used

- Tier 3 agent details were not available in this report.

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- No Tier 3 evaluation signal details were available in this report.

## Test Tasks

Tier 3 evaluation task details were not available in this report.

## Results

Tier 3 dimension rollup was not available in this report.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 19 total findings.

Top findings:

- MEDIUM QUALITY/quality_efficiency: Deeply nested references in performance-gotchas.md (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)
- MEDIUM SECURITY/Unknown (SQP-2): The skill outputs shell commands and Python source files with Triton kernel code, but the skill card does not include ex (`skill-card.md:34`)
- LOW QUALITY/quality_discoverability: Description very long (505 chars, recommend 50-150) (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)
- LOW QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/tilegym-converting-cutile-to-triton/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 4 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within translations/workflow.md:
  "### TMA Setup (Required Once)" in translations/workflow.md (lines 208-218)
  vs "# TMA allocator (required once per kernel launch context)" in translations/workflow.md (lines 362-368) (`translations/workflow.md:208`)
- HIGH DUPLICATE/duplicate: Duplicate content found across references/harness-integration.md and translations/workflow.md:
  "# Testing & Validation (cuTile → Triton)" in references/harness-integration.md (lines 1-7)
  vs "# Performance testing (Triton vs cuTile)" in translations/workflow.md (lines 168-170)
  vs "### Step 1: Benchmark" in translations/workflow.md (lines 236-243) (`references/harness-integration.md:1`)
- HIGH DUPLICATE/duplicate: Duplicate content found within translations/workflow.md:
  "## TMA OPTIMIZATION (Phase c2t-4) {#tma-optimization-phase-c2t-4}" in translations/workflow.md (lines 178-181)
  vs "### Performance Killer #1: Raw Pointer Arithmetic vs TMA Tensor Descriptors" in translations/workflow.md (lines 329-335) (`translations/workflow.md:178`)
- HIGH DUPLICATE/duplicate: Duplicate content found within translations/workflow.md:
  "### Triton Debug / Profiling" in translations/workflow.md (lines 115-125)
  vs "# Triton profiling / autotune visibility" in translations/workflow.md (lines 171-177) (`translations/workflow.md:115`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
