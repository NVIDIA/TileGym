# Evaluation Report

Evaluation of the `tilegym-cutile-python` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `tilegym-cutile-python`
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

Tier 1 validation reported findings. NVSkills-Eval ran 9 checks and found 11 total findings.

Top findings:

- LOW QUALITY/quality_discoverability: Description very long (222 chars, recommend 50-150) (`skills/tilegym-cutile-python/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/tilegym-cutile-python/SKILL.md`)
- LOW QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/tilegym-cutile-python/SKILL.md`)
- LOW QUALITY/quality_reliability: No limitations documented (`skills/tilegym-cutile-python/SKILL.md`)
- LOW QUALITY/quality_efficiency: Uses complex/corporate language (`skills/tilegym-cutile-python/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 14 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across examples/convolution/conv2d_with_bias_dilation_groups.py and examples/convolution/conv3d_with_bias_dilation_groups.py and examples/convolution/conv_transpose_2d.py and examples/convolution/conv_transpose_3d.py and examples/matmul/matmul_4d_tensors.py and examples/matmul/split_k_gemm.py:
  "_adjust_group_size()" in examples/convolution/conv2d_with_bias_dilation_groups.py (lines 39-44)
  vs "_adjust_group_size()" in examples/convolution/conv3d_with_bias_dilation_groups.py (lines 42-47)
  vs "_adjust_group_size()" in examples/convolution/conv_transpose_2d.py (lines 48-53)
  vs "_adjust_group_size()" in examples/convolution/conv_transpose_3d.py (lines 49-54)
  vs "_adjust_group_size()" in examples/matmul/matmul_4d_tensors.py (lines 36-41)
  vs "_adjust_group_size()" in examples/matmul/split_k_gemm.py (lines 21-26) (`examples/convolution/conv2d_with_bias_dilation_groups.py:39`)
- HIGH DUPLICATE/duplicate: Duplicate content found across examples/convolution/conv2d_with_bias_dilation_groups.py and examples/convolution/conv3d_with_bias_dilation_groups.py and examples/convolution/conv_transpose_2d.py and examples/convolution/conv_transpose_3d.py:
  "_select_tile_config_2d()" in examples/convolution/conv2d_with_bias_dilation_groups.py (lines 47-87)
  vs "_select_tile_config_3d()" in examples/convolution/conv3d_with_bias_dilation_groups.py (lines 50-88)
  vs "_select_tile_config_trans2d()" in examples/convolution/conv_transpose_2d.py (lines 56-94)
  vs "_select_tile_config_trans3d()" in examples/convolution/conv_transpose_3d.py (lines 57-95) (`examples/convolution/conv2d_with_bias_dilation_groups.py:47`)
- HIGH DUPLICATE/duplicate: Duplicate content found across examples/matmul/matmul_4d_tensors.py and examples/matmul/matrix_vector_multiplication.py and examples/matmul/split_k_gemm.py:
  "reference_matmul()" in examples/matmul/matmul_4d_tensors.py (lines 101-103)
  vs "reference_matmul()" in examples/matmul/matrix_vector_multiplication.py (lines 54-56)
  vs "reference_gemm()" in examples/matmul/split_k_gemm.py (lines 129-131) (`examples/matmul/matmul_4d_tensors.py:101`)
- HIGH DUPLICATE/duplicate: Duplicate content found across examples/convolution/conv2d_with_bias_dilation_groups.py and examples/convolution/conv3d_with_bias_dilation_groups.py and examples/convolution/conv_transpose_2d.py and examples/convolution/conv_transpose_3d.py and orchestration/composer_agent.md:
  "pytorch_reference()" in examples/convolution/conv2d_with_bias_dilation_groups.py (lines 305-307)
  vs "pytorch_reference()" in examples/convolution/conv3d_with_bias_dilation_groups.py (lines 329-331)
  vs "pytorch_reference()" in examples/convolution/conv_transpose_2d.py (lines 305-308)
  vs "pytorch_reference()" in examples/convolution/conv_transpose_3d.py (lines 336-338)
  vs "# ============================================================" in orchestration/composer_agent.md (lines 100-105) (`examples/convolution/conv2d_with_bias_dilation_groups.py:305`)
- HIGH DUPLICATE/duplicate: Duplicate content found within orchestration/composer_agent.md:
  "# ============================================================" in orchestration/composer_agent.md (lines 64-71)
  vs "# ============================================================" in orchestration/composer_agent.md (lines 74-81) (`orchestration/composer_agent.md:64`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
