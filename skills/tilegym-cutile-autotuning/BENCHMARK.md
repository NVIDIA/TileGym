# Evaluation Report

Evaluation of the `tilegym-cutile-autotuning` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `tilegym-cutile-autotuning`
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

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 18 total findings.

Top findings:

- MEDIUM PII/phone_numbers: International phone number (`SKILL.md:206`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/tilegym-cutile-autotuning/SKILL.md`)
- MEDIUM QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/tilegym-cutile-autotuning/SKILL.md`)
- MEDIUM QUALITY/quality_efficiency: Deeply nested references in workflow.md (`skills/tilegym-cutile-autotuning/SKILL.md`)
- MEDIUM SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/tilegym-cutile-autotuning/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 3 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found within references/search-strategies.md:
  "# 2. Tune once (exhaustive search over all configs)" in references/search-strategies.md (lines 19-29)
  vs "# Step 1: Run exhaustive_search to find optimal config (outside NCU)" in references/search-strategies.md (lines 100-104) (`references/search-strategies.md:19`)
- HIGH DUPLICATE/duplicate: Duplicate content found across assets/examples/03_rope_inplace_splitbuffer/autotuned_launch.py and assets/examples/03_rope_inplace_splitbuffer/fixed_launch.py:
  "precompute_freqs()" in assets/examples/03_rope_inplace_splitbuffer/autotuned_launch.py (lines 112-117)
  vs "precompute_freqs()" in assets/examples/03_rope_inplace_splitbuffer/fixed_launch.py (lines 89-95) (`assets/examples/03_rope_inplace_splitbuffer/autotuned_launch.py:112`)
- HIGH DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "# Module-level cache: tune once, launch fast forever after" in SKILL.md (lines 47-59)
  vs "# Module-level cache: tune once, launch fast forever after" in SKILL.md (lines 60-63) (`SKILL.md:47`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
