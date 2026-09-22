# Skill Benchmark: tilegym-cutile-optimizing

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tilegym-cutile-optimizing`
- Evaluation date: 2026-09-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (2 positive, 3 negative)
- Dataset digest: `sha256:2bd0b05ee7d03ff8467e82b63e748fb84dbd6dc4c7f48a2c9c61bc3940ffa8f5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 84.0% → 100.0% (+16.0 points) | 68.6% → 96.0% (+27.4 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 78.7% → 94.7% (+16.0 points) | 60.5% → 91.7% (+31.2 points) |
| Efficiency | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 97.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 583,454 | 279,417 | +304,037 | +108.81% | skill 5/5; base 5/5 |
| claude-code | tilegym-cutile-optimizing-001 | 196,520 | 153,062 | +43,458 | +28.39% | skill 1/1; base 1/1 |
| claude-code | tilegym-cutile-optimizing-002 | 291,482 | 34,513 | +256,969 | +744.56% | skill 1/1; base 1/1 |
| claude-code | tilegym-cutile-optimizing-003-oauth-pkce-negative | 32,063 | 31,547 | +516 | +1.64% | skill 1/1; base 1/1 |
| claude-code | tilegym-cutile-optimizing-004-postgres-autovacuum-negative | 31,504 | 29,023 | +2,481 | +8.55% | skill 1/1; base 1/1 |
| claude-code | tilegym-cutile-optimizing-005-websocket-backoff-negative | 31,885 | 31,272 | +613 | +1.96% | skill 1/1; base 1/1 |
| codex | All cases | 211,209 | 341,074 | N/A | N/A | skill 5/5; base 7/7 |
| codex | tilegym-cutile-optimizing-001 | 83,424 | 216,552 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tilegym-cutile-optimizing-002 | 63,840 | 23,408 | +40,432 | +172.73% | skill 1/1; base 1/1 |
| codex | tilegym-cutile-optimizing-003-oauth-pkce-negative | 18,814 | 18,296 | +518 | +2.83% | skill 1/1; base 1/1 |
| codex | tilegym-cutile-optimizing-004-postgres-autovacuum-negative | 30,370 | 68,613 | -38,243 | -55.74% | skill 1/1; base 1/1 |
| codex | tilegym-cutile-optimizing-005-websocket-backoff-negative | 14,761 | 14,205 | +556 | +3.91% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 794,663 | 620,491 | N/A | N/A | skill 10/10; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/tilegym-cutile-optimizing/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/tilegym-cutile-optimizing/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (541 chars, recommend 50-150) (`skills/tilegym-cutile-optimizing/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/tilegym-cutile-optimizing/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/tilegym-cutile-optimizing/SKILL.md`)
- 3 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
