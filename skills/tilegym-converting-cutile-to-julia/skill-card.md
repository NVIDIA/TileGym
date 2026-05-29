## Description: <br>
Converts cuTile Python GPU kernels (@ct.kernel) to cuTile.jl Julia equivalents, handling kernel syntax translation, 0-indexed to 1-indexed conversion, broadcasting differences, memory layout (row-major to column-major), type system mapping, and launch API differences. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to port cuTile Python GPU kernels to Julia cuTile.jl equivalents, enabling Julia-native GPU kernel development without a Python bridge. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [API Mapping (Python to Julia)](references/api-mapping.md) <br>
- [Critical Rules](references/critical-rules.md) <br>
- [Debugging Guide](references/debugging.md) <br>
- [Testing & Verification Guide](references/testing.md) <br>
- [Conversion Workflow](translations/workflow.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands] <br>
**Output Format:** [Julia source files (.jl) with inline documentation] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
v1.3.0 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
