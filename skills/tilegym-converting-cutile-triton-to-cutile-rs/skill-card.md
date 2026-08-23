## Description: <br>
Use this skill to convert, port, or translate Triton-TileIR or cuTile-Python GPU kernels to cutile-rs (Rust) via a bounded multi-agent pipeline that delegates kernel, host, correctness, and performance work to specialized sub-agents. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to port Triton-TileIR or cuTile-Python GPU kernels to cutile-rs (Rust), enabling tile-based GPU programming with a Rust backend. <br>

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
- [op-mapping.md](references/op-mapping.md) <br>
- [output-structure.md](references/output-structure.md) <br>
- [performance-checklist.md](references/performance-checklist.md) <br>
- [pipeline.md](references/pipeline.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands] <br>
**Output Format:** [Rust source files and Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (1 positive, 3 negative) from dataset skill-evaluator-dataset-snapshot/1, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83% → 99% (+16 points) | 79% → 98% (+18 points) |
| Security | 100% → 100% (±0 points) | 75% → 100% (+25 points) |
| Correctness | 75% → 100% (+25 points) | 75% → 100% (+25 points) |
| Discoverability | 88% → 100% (+12 points) | 86% → 92% (+6 points) |
| Effectiveness | 79% → 100% (+21 points) | 78% → 98% (+20 points) |
| Efficiency | 75% → 96% (+21 points) | 84% → 100% (+16 points) |

## Skill Version(s): <br>
d59cb2d (source: git SHA, committed 2026-08-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
