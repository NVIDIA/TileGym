## Description: <br>
Question-gated router for the curated cuTile optimization wiki: kernel-family playbooks, measured techniques with caveats, performance patterns, and practical language knowledge for Blackwell and Hopper. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers optimizing cuTile GPU kernels use this skill to consult a curated knowledge wiki for kernel-family playbooks, measured optimization techniques, and cuTile language details targeting Blackwell and Hopper architectures. <br>

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
- [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python) <br>
- [cuTile Python Source](https://github.com/NVIDIA/cutile-python) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (2 positive, 3 negative), each with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.4% | 95.1% |
| Security | 100.0% → 100.0% (±0.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 84.0% → 100.0% (+16.0 pts) | 68.6% → 96.0% (+27.4 pts) |
| Discoverability | 97.5% | 90.0% |
| Effectiveness | 78.7% → 94.7% (+16.0 pts) | 60.5% → 91.7% (+31.2 pts) |
| Efficiency | 89.8% | 97.7% |

## Skill Version(s): <br>
4cdde65 (source: git SHA, committed 2026-09-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
