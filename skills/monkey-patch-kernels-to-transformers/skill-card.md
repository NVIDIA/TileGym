## Description: <br>
Integrate TileGym kernels into Hugging Face `transformers` models by replacing the library's submodule(s) and certain class(es)' implementations, and patching certain class(es)' init/forward/load weight methods prior to instantiating models. <br>

This skill is for research and development only. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to integrate TileGym GPU kernels into Hugging Face transformers models using a non-intrusive monkey-patch approach to validate end-to-end functional correctness and improve performance. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Environment Setup](references/environment-setup.md) <br>
- [Kernel Integration Workflow](references/kernel-integration.md) <br>
- [Auto Kernelize](references/auto-kernelize.md) <br>
- [Workflow Diagram](references/workflow-diagram.png) <br>
- [CUDA Tile IR Supported Architectures](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#supported-architectures) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
2026.05.05-beta (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
