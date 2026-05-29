## Description: <br>
Expert cuTile programming assistant that writes high-performance GPU kernels using cuTile's tile-based programming model with proper validation, optimization, and deep agent orchestration for complex multi-kernel tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to write, debug, and optimize high-performance GPU kernels using cuTile's tile-based programming model, including complex multi-kernel workflows via orchestrated sub-agents. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuTile Language Specification](https://docs.nvidia.com/cuda/cutile-python) <br>
- [Implementation Lessons](guidelines/01_implementation_lessons.md) <br>
- [Code Generation Rules](guidelines/02_code_generation_rules.md) <br>
- [Core Concepts](guidelines/03_concepts.md) <br>
- [Orchestration Overview](orchestration/overview.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands] <br>
**Output Format:** [Python source files with inline validation] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
Evaluated via NVSkills-Eval 3-Tier framework (Tier 1: 9 static validation checks, Tier 2: 2 deduplication checks). Tier 3 live agent evaluation not available in this report. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
1.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
