## Description: <br>
Converts cuTile GPU kernels (@ct.kernel) to Triton (@triton.jit), handling standard in-repo conversion, debugging, and mapping cuTile idioms to Triton equivalents including dual-kernel layout flags and TMA optimization. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to convert cuTile GPU kernels to Triton, porting @ct.kernel functions to @triton.jit equivalents with correct TMA usage, debugging, and performance validation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [API Mapping (cuTile to Triton)](references/api-mapping.md) <br>
- [Debugging Guide](references/debugging.md) <br>
- [Common Translation Gotchas](references/gotchas.md) <br>
- [Harness Integration](references/harness-integration.md) <br>
- [Optimization Strategy](references/optimization-strategy.md) <br>
- [Optimizing Reference (GEMM/BMM/Attention)](references/optimizing-reference.md) <br>
- [Performance Gotchas](references/performance-gotchas.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands] <br>
**Output Format:** [Python source files and shell command output] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
Evaluated through NVSkills-Eval 3-Tier framework (external profile). Tier 1 static validation ran 9 checks with 19 findings. Tier 2 deduplication ran 2 checks with 4 findings. Tier 3 live agent evaluation not available. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
