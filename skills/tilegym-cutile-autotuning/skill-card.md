## Description: <br>
Use when adding, modifying, optimizing, or debugging CuTile autotuning code. Trigger signals: `exhaustive_search` / `replace_hints` / `hints_fn` / `cuda.tile.tune` in code, `autotune` in filenames, or correctness/performance issues in autotuned CuTile kernels. Covers: tune-once/cache/launch pattern, per-architecture configs (sm80–sm120), parameter space design (tile sizes, occupancy, num_ctas), and 7 common pitfalls with solutions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with CuTile GPU kernels use this skill to add, optimize, or debug autotuning configurations for CUDA Tile kernels across NVIDIA GPU architectures (sm80–sm120). <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [exhaustive_search API Reference](references/api-reference.md) <br>
- [Hardware Constraints](references/hardware-constraints.md) <br>
- [Kernel Type Templates](references/kernel-type-templates.md) <br>
- [Parameter Space Design](references/parameter-space-design.md) <br>
- [Pitfalls](references/pitfalls.md) <br>
- [Search Strategies](references/search-strategies.md) <br>
- [Workflow](references/workflow.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
v1.3.0 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
