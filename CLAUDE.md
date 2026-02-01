<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TileGym is a CUDA Tile kernel library providing GPU kernel tutorials and examples for tile-based programming on NVIDIA Blackwell GPUs (B200, RTX 5080, RTX 5090). It includes kernel implementations for deep learning operators and end-to-end LLM integrations (Llama 3.1, DeepSeek V2, Qwen2).

**Requirements**: CUDA 13.1, PyTorch 2.9.1+, Blackwell architecture GPU

## Build and Install

```bash
# Install (requires torch and triton pre-installed)
pip install .

# Development mode
pip install -e .

# Install dev dependencies (ruff for linting)
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run a specific op test
pytest tests/ops/test_<op_name>.py -k test_op -v --log-cli-level=INFO

# Example: test RMSNorm
pytest tests/ops/test_rms_norm.py -k test_op -v --log-cli-level=INFO
```

## Running Benchmarks

```bash
# Run all benchmarks
cd tests/benchmark && bash run_all.sh

# Run single benchmark
python tests/benchmark/bench_matrix_multiplication.py
```

## Linting

```bash
ruff check .
ruff format .
```

Ruff config: 120 char line length, isort-style import sorting (force-single-line)

## Architecture

### Backend Dispatch System

TileGym uses a backend dispatch pattern (`src/tilegym/backend/`) that routes operation calls to different implementations:

- **dispatcher.py**: `@dispatch(name)` decorator wraps operations; `@register_impl(name, backend)` registers backend-specific implementations
- **selector.py**: `set_backend()` / `get_current_backend()` controls active backend
- Backends: `cutile` (CUDA Tile), `pytorch` (fallback)
- Set `DISABLE_FALLBACK=1` env var to error instead of falling back to PyTorch

### Operations (`src/tilegym/ops/`)

- **ops.py**: Unified interface defining all operations with `@dispatch` decorators
- **cutile/**: CUDA Tile kernel implementations (matmul, attention, rope, rmsnorm, mla, softmax, etc.)
  - Most ops use `torch.autograd.Function` subclasses for backward pass support
  - Naming pattern: `*Function` classes (e.g., `SiLUMulFunction`, `RMSNorm`, `TileRopeFunction`)
- **attn_interface.py**, **moe_interface.py**: Higher-level interfaces for attention and MoE
- **fused_swiglu.py**: Fused backward pass implementation for SwiGLU (gate + up projection)

Key operations: `matmul`, `bmm`, `group_gemm`, `rms_norm`, `softmax`, `fmha`, `fmha_decode`, `mla`, `mla_decoding`, `apply_rope_base`, `silu_and_mul`, `swiglu`, `dropout`

### Kernel Implementation Pattern

CUDA Tile kernels follow a consistent structure:
1. `@ct.kernel` decorated forward/backward kernel functions
2. Python wrapper functions (e.g., `swiglu_forward`, `swiglu_backward`)
3. `torch.autograd.Function` subclass connecting forward/backward
4. `@register_impl` decorator to register with backend dispatcher

### Transformers Integration (`src/tilegym/transformers/`)

- **monkey_patch.py**: Patches HuggingFace transformers to use TileGym kernels
- **deepseek2/**: DeepSeek V2 model implementation

### Test Structure (`tests/`)

- **tests/ops/**: Functional correctness tests using `common.PyTestCase`
- **tests/benchmark/**: Performance micro-benchmarks
- **tests/common.py**: Base `PyTestCase` class with `assertCorrectness()` for comparing TileGym vs reference implementations

Test naming conventions:
- Classes: `Test_<OpName>` (e.g., `Test_RoPE`, `Test_SwiGLU`)
- Functional correctness: `test_op` method
- Backward pass tests: often separate test methods or files (e.g., `test_fused_swiglu_backward.py`)
- Tests parametrized across backends, shapes, and dtypes

## Adding New Operations

When implementing a new operation with backward pass support:

1. **Create kernel functions**: Define `@ct.kernel` decorated functions for forward and backward passes
2. **Add wrapper functions**: Write Python wrappers that handle shape reshaping and kernel launch
3. **Implement autograd.Function**: Subclass `torch.autograd.Function` with `forward()` and `backward()` methods
   - Use `ctx.save_for_backward()` to save tensors needed in backward pass
4. **Register with dispatcher**: Use `@register_impl(name, backend="cutile")` decorator
5. **Add to ops.py**: Define `@dispatch(name)` decorated interface function
6. **Write tests**: Add test file in `tests/ops/` with correctness tests comparing against reference implementation
   - Use `self.assertCorrectness()` from `PyTestCase` to compare outputs
   - Test both forward and backward passes if applicable

See `src/tilegym/ops/cutile/swiglu.py` for a complete example.

## LLM Inference Examples

```bash
# Baseline (no TileGym)
python modeling/transformers/infer.py --model_id meta-llama/Meta-Llama-3.1-8B --show_outputs

# With TileGym CUTILE backend
python modeling/transformers/infer.py --model_id meta-llama/Meta-Llama-3.1-8B --use_tilegym --use_cutile --use_attn --show_outputs
```

Docker available: `docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .`
