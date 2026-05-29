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

- HIGH DUPLICATE/duplicate: Duplicate content found across examples/convolution/conv2d_with_bias_dilation_groups.py and examples/convolution/conv3d_with_bias_dilation_groups.py and examples/convolution/conv_transpose_2d.py and examples/convolution/conv_transpose_3d.py:
  "_select_tile_config_2d()" in examples/convolution/conv2d_with_bias_dilation_groups.py (lines 47-87)
  vs "_select_tile_config_3d()" in examples/convolution/conv3d_with_bias_dilation_groups.py (lines 50-88)
  vs "_select_tile_config_trans2d()" in examples/convolution/conv_transpose_2d.py (lines 56-94)
  vs "_select_tile_config_trans3d()" in examples/convolution/conv_transpose_3d.py (lines 57-95) (`examples/convolution/conv2d_with_bias_dilation_groups.py:47`)
- HIGH DUPLICATE/duplicate: Duplicate content found across examples/matmul/matmul_4d_tensors.py and examples/matmul/matrix_vector_multiplication.py and examples/matmul/split_k_gemm.py:
  "reference_matmul()" in examples/matmul/matmul_4d_tensors.py (lines 101-103)
  vs "reference_matmul()" in examples/matmul/matrix_vector_multiplication.py (lines 54-56)
  vs "reference_gemm()" in examples/matmul/split_k_gemm.py (lines 129-131) (`examples/matmul/matmul_4d_tensors.py:101`)
- HIGH DUPLICATE/duplicate: Duplicate content found across examples/convolution/conv2d_with_bias_dilation_groups.py and examples/convolution/conv3d_with_bias_dilation_groups.py and examples/convolution/conv_transpose_2d.py and examples/convolution/conv_transpose_3d.py and examples/matmul/matmul_4d_tensors.py and examples/matmul/split_k_gemm.py:
  "_adjust_group_size()" in examples/convolution/conv2d_with_bias_dilation_groups.py (lines 39-44)
  vs "_adjust_group_size()" in examples/convolution/conv3d_with_bias_dilation_groups.py (lines 42-47)
  vs "_adjust_group_size()" in examples/convolution/conv_transpose_2d.py (lines 48-53)
  vs "_adjust_group_size()" in examples/convolution/conv_transpose_3d.py (lines 49-54)
  vs "_adjust_group_size()" in examples/matmul/matmul_4d_tensors.py (lines 36-41)
  vs "_adjust_group_size()" in examples/matmul/split_k_gemm.py (lines 21-26) (`examples/convolution/conv2d_with_bias_dilation_groups.py:39`)
- HIGH DUPLICATE/duplicate: Duplicate content found within orchestration/composer_agent.md:
  "# ============================================================" in orchestration/composer_agent.md (lines 64-71)
  vs "# ============================================================" in orchestration/composer_agent.md (lines 74-81) (`orchestration/composer_agent.md:64`)
- HIGH DUPLICATE/duplicate: Duplicate content found across torch-learner/references/1_pytorch_codebase_map.md and torch-learner/references/2_dispatch_mechanism.md and torch-learner/references/3_tracing_strategies.md and torch-learner/references/4_language_layers.md and torch-learner/tracing_workflow.md:
  "# Find a function in the functional API" in torch-learner/references/1_pytorch_codebase_map.md (lines 44-49)
  vs "# Python reference implementations of ATen ops" in torch-learner/references/1_pytorch_codebase_map.md (lines 71-73)
  vs "# The master registry of ALL ATen operations — find it:" in torch-learner/references/1_pytorch_codebase_map.md (lines 93-94)
  vs "### Finding C++ Op Implementations" in torch-learner/references/1_pytorch_codebase_map.md (lines 133-137)
  vs "# Step 1: Find the op in native_functions.yaml" in torch-learner/references/1_pytorch_codebase_map.md (lines 138-140)
  vs "# Step 3: Search for those function names in the C++ source" in torch-learner/references/1_pytorch_codebase_map.md (lines 144-147)
  vs "# Search all .cpp files under native/" in torch-learner/references/1_pytorch_codebase_map.md (lines 151-153)
  vs "# Search CUDA kernels (.cu files)" in torch-learner/references/1_pytorch_codebase_map.md (lines 154-156)
  vs "# Search cuDNN wrappers" in torch-learner/references/1_pytorch_codebase_map.md (lines 157-159)
  vs "# List CUDA kernel files" in torch-learner/references/1_pytorch_codebase_map.md (lines 179-182)
  vs "# 2. Python functional" in torch-learner/references/1_pytorch_codebase_map.md (lines 214-216)
  vs "# 3. native_functions.yaml entry" in torch-learner/references/1_pytorch_codebase_map.md (lines 217-219)
  vs "# 4. C++ implementation (follow function names from YAML dispatch table)" in torch-learner/references/1_pytorch_codebase_map.md (lines 220-222)
  vs "# Find cuDNN calls for an op" in torch-learner/references/1_pytorch_codebase_map.md (lines 230-232)
  vs "# Find cuBLAS usage" in torch-learner/references/1_pytorch_codebase_map.md (lines 233-235)
  vs "# Find CUDA kernel launches" in torch-learner/references/1_pytorch_codebase_map.md (lines 236-239)
  vs "# Check what the Python function calls" in torch-learner/references/1_pytorch_codebase_map.md (lines 243-245)
  vs "# Look for torch._C calls" in torch-learner/references/1_pytorch_codebase_map.md (lines 249-251)
  vs "# Look for torch.ops calls" in torch-learner/references/1_pytorch_codebase_map.md (lines 252-255)
  vs "# Check Python reference implementations" in torch-learner/references/1_pytorch_codebase_map.md (lines 262-265)
  vs "## native_functions.yaml: The Master Registry" in torch-learner/references/2_dispatch_mechanism.md (lines 38-41)
  vs "# Calling ops by their registered name:" in torch-learner/references/2_dispatch_mechanism.md (lines 196-198)
  vs "## Tracing an Op Through Dispatch: Quick Reference" in torch-learner/references/2_dispatch_mechanism.md (lines 273-281)
  vs "### For CPU implementations" in torch-learner/references/3_tracing_strategies.md (lines 184-195)
  vs "### For CUDA implementations" in torch-learner/references/3_tracing_strategies.md (lines 196-214)
  vs "### Finding cuDNN-accelerated operations" in torch-learner/references/3_tracing_strategies.md (lines 215-219)
  vs "# Search for cuDNN wrappers related to your op" in torch-learner/references/3_tracing_strategies.md (lines 220-222)
  vs "# List all cuDNN wrapper files in your PyTorch version" in torch-learner/references/3_tracing_strategies.md (lines 223-226)
  vs "# Direct access to registered C++ operators" in torch-learner/references/4_language_layers.md (lines 66-72)
  vs "# Find a functional function" in torch-learner/references/4_language_layers.md (lines 358-360)
  vs "# Find in native_functions.yaml" in torch-learner/references/4_language_layers.md (lines 367-369)
  vs "# Find cuDNN usage" in torch-learner/references/4_language_layers.md (lines 382-384)
  vs "# Find cuBLAS usage" in torch-learner/references/4_language_layers.md (lines 385-388)
  vs "### Step 5: Look Up native_functions.yaml" in torch-learner/tracing_workflow.md (lines 90-108)
  vs "### Step 6: Find C++ and Device-Specific Implementations" in torch-learner/tracing_workflow.md (lines 109-113)
  vs "# C++ implementation (.cpp and .cu files)" in torch-learner/tracing_workflow.md (lines 114-116)
  vs "# cuDNN wrappers" in torch-learner/tracing_workflow.md (lines 120-122)
  vs "# cuBLAS (for linear algebra ops)" in torch-learner/tracing_workflow.md (lines 123-128) (`torch-learner/references/1_pytorch_codebase_map.md:44`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
