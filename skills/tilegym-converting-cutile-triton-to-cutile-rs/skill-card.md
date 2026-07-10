## Description: <br>
Orchestrates a multi-agent pipeline to convert Triton-TileIR (nvtriton) or cuTile-Python GPU kernels to cutile-rs (Rust), delegating kernel writing, compilation, correctness testing, and performance validation to specialized sub-agents. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to automatically port Triton-TileIR or cuTile-Python GPU kernels to the cutile-rs Rust backend, reducing manual translation effort and ensuring correctness through automated validation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [coding-rules.md](references/coding-rules.md) <br>
- [env-block.md](references/env-block.md) <br>
- [ir-diff-checklist.md](references/ir-diff-checklist.md) <br>
- [no-mut-tensor-output.md](references/no-mut-tensor-output.md) <br>
- [op-mapping.md](references/op-mapping.md) <br>
- [output-structure.md](references/output-structure.md) <br>
- [performance-checklist.md](references/performance-checklist.md) <br>
- [pipeline.md](references/pipeline.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Analysis] <br>
**Output Format:** [Rust source files, Python wrappers, MLIR IR, and Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
4 evaluation tasks (1 positive skill-activation, 3 negative) via NVSkills-Eval external profile in astra-sandbox environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 4 | 100% (+0%) | 100% (+0%) |
| Correctness | 4 | 86% (+21%) | 99% (+15%) |
| Discoverability | 4 | 81% (+6%) | 99% (+13%) |
| Effectiveness | 4 | 83% (+25%) | 95% (+24%) |
| Efficiency | 4 | 76% (-2%) | 91% (+12%) |

## Skill Version(s): <br>
v1.3.0-66-g4d4e968 (source: git describe) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
