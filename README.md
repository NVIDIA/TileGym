<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

English | [简体中文](README_chs.md) | [繁體中文](README_cht.md) | [日本語](README_ja.md) | [Français](README_fr.md)

# TileGym

TileGym is a CUDA Tile kernel library that provides a rich collection of kernel tutorials and examples for tile-based GPU programming.

[**Overview**](#overview) |
[**Features**](#features) |
[**Installation**](#installation) |
[**Quick Start**](#quick-start) |
[**Contributing**](#contributing) |
[**License**](#license-and-third-party-notices)

## Overview

This repository aims to provide helpful kernel tutorials and examples for tile-based GPU programming. TileGym is a playground for experimenting with CUDA Tile, where you can learn how to build efficient GPU kernels and explore their integration into real-world large language models such as Llama 3.1 and DeepSeek V2. Whether you're learning tile-based GPU programming or looking to optimize your LLM implementations, TileGym offers practical examples and comprehensive guidance.
<img width="95%" alt="tilegym_1_newyear" src="https://github.com/user-attachments/assets/f37010f5-14bc-44cd-bddf-f517dc9922b8" />

## Features

- Rich collection of CUDA Tile kernel examples
- Practical kernel implementations for common deep learning operators
- Performance benchmarking to evaluate kernel efficiency
- End-to-end integration examples with popular LLMs (Llama 3.1, DeepSeek V2)

## Installation

### Prerequisites

> **GPU Support**: TileGym requires **CUDA 13.1+** and a **Blackwell GPU** (e.g., B200, RTX 5080, RTX 5090). **NVIDIA Ampere** (e.g., A100) is also supported with **CUDA 13.2+**. All released kernels are validated on both architectures. Download CUDA from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

- PyTorch (version 2.9.1 or compatible)
- **[CUDA 13.1+](https://developer.nvidia.com/cuda-downloads)** (Required - TileGym is built and tested exclusively on CUDA 13.1+)
- Triton (included with PyTorch installation)

### Setup Steps

#### 1. Prepare `torch` and `triton` environment

If you already have `torch` and `triton`, skip this step.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

We have verified that `torch==2.9.1` works. You can also get `triton` packages when installing `torch`.

#### 2. Install TileGym

TileGym uses [`cuda-tile`](https://github.com/nvidia/cutile-python) (≥ 1.3.0) for GPU kernel programming, which depends on the `tileiras` compiler at runtime.

##### Install from PyPI (recommended)

```bash
pip install tilegym[tileiras]
```

This installs TileGym and all runtime dependencies, including `cuda-tile[tileiras]` which bundles the `tileiras` compiler directly into your Python environment.

If you already have `tileiras` available on your system (e.g., from [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads)), you can omit the extra:

```bash
pip install tilegym
```

##### Install from source

```bash
git clone https://github.com/NVIDIA/TileGym.git
cd TileGym
pip install .[tileiras]   # or: pip install .  (if you have system tileiras)
```

For editable (development) mode, use `pip install -e .` or `pip install -e .[tileiras]`.

All runtime dependencies are declared in [`requirements.txt`](requirements.txt) and are installed automatically by both `pip install tilegym` and `pip install .`.

We also provide Dockerfile, you can refer to [modeling/transformers/README.md](modeling/transformers/README.md).

### Backends

TileGym provides kernels for the following backends, each in its own folder under `src/tilegym/ops/`:

- **cuTile** (default) — [`src/tilegym/ops/cutile`](src/tilegym/ops/cutile), see more details in [cutile-python](https://github.com/nvidia/cutile-python).
- **CUDA Tile C++** — [`src/tilegym/ops/tilecpp`](src/tilegym/ops/tilecpp), see more details in [README.tilecpp.md](README.tilecpp.md).
- **Triton CUDA Tile IR** — [`src/tilegym/ops/triton`](src/tilegym/ops/triton), see more details in [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR).

To use the Triton CUDA Tile IR backend, install its wheel into a separate directory and select it at runtime with `ENABLE_TILE=1`. Wheels for CPython 3.12 and 3.13 are available on the [releases page](https://github.com/triton-lang/Triton-to-tile-IR/releases):

```bash
# Install into a separate directory, kept apart from the default environment
pip install --target /opt/nvtriton <nvtriton-wheel-for-your-python>.whl

# Select the Triton CUDA Tile IR backend at runtime
PYTHONPATH=/opt/nvtriton ENABLE_TILE=1 python your_script.py
```

## Quick Start

There are three main ways to use TileGym:

### 1. Explore Kernel Examples

All kernel implementations are located in the `src/tilegym/ops/` directory. You can test individual operations with minimal scripts. Function-level usage and minimal scripts for individual ops are documented in [tests/ops/README.md](tests/ops/README.md)

### 2. Run Benchmarks

Evaluate kernel performance with micro-benchmarks:

```bash
cd tests/benchmark
bash run_all.sh
```

Complete benchmark guide available in [tests/benchmark/README.md](tests/benchmark/README.md)

### 3. Run LLM Transformer Examples

Use TileGym kernels in end-to-end inference scenarios. We provide runnable scripts and instructions for transformer language models (e.g., Llama 3.1-8B) accelerated using TileGym kernels.

First, install the additional dependency:

```bash
pip install accelerate==1.13.0 --no-deps
```

**Containerized Setup (Docker)**:

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

More details in [modeling/transformers/README.md](modeling/transformers/README.md)

### 4. Julia (cuTile.jl) Kernels (Optional)

TileGym also includes experimental [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) kernel implementations in Julia. These are self-contained in the `julia/` directory and do not require the Python TileGym package.

**Prerequisites**: [Julia 1.12+](https://julialang.org/downloads/), CUDA 13.1, Blackwell GPU

```bash
# Install Julia (if not already installed)
curl -fsSL https://install.julialang.org | sh

# Install dependencies
julia --project=julia/ -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=julia/ julia/test/runtests.jl
```

See `julia/Project.toml` for the full dependency list.

### 5. Enable the cuTile-rs (Rust) backend (Optional)

A subset of ops ships an additional **cuTile-rs** backend under
[`src/tilegym/ops/cutile_rs`](src/tilegym/ops/cutile_rs) — kernels authored in
Rust with [`cutile-rs`](https://github.com/NVlabs/cutile-rs) and loaded through
a C-ABI `libcutile_kernels.so`. It is opt-in and only usable from a source
checkout.

**Prerequisites** (in addition to the base install above), matching
[cuTile-rs](https://github.com/NVlabs/cutile-rs):

- **Rust 1.89+** — `cargo` and `rustc` on `PATH`:

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup default stable
  ```

- **CUDA toolkit with headers** — the Rust build runs `bindgen` against
  `cuda.h`. Set `CUDA_TOOLKIT_PATH` to your install; if unset, cuTile-rs falls
  back to `/usr/local/cuda`:

  ```bash
  export CUDA_TOOLKIT_PATH=/usr/local/cuda   # must contain include/cuda.h
  ```

**Use it.** The backend loader builds the shared library lazily on first use
(`cargo build --release`), so no manual build step is required:

```python
import tilegym
tilegym.set_backend("cutile-rs")

from tilegym.backend.selector import get_available_backends
print(get_available_backends())        # should include "cutile-rs"

from tilegym.ops import bmm             # backend-agnostic import
# ... bmm(...) now dispatches to the cuTile-rs kernel
```

**Optional environment knobs:**

```bash
export CUTILE_RS_AUTOBUILD=0                          # skip the lazy rebuild; use a prebuilt .so
export CUTILE_RS_KERNELS_DIR=/abs/path/to/cutile_kernels   # override the crate location
```

> If `cargo` is not on `PATH` and no prebuilt `libcutile_kernels.so` is present,
> the backend reports itself unavailable and cuTile-rs tests are skipped rather
> than failing.

**Benchmarking cuTile-rs.** When comparing cuTile-rs perf against the
cuTile-Python baseline, run the perf tests with **`CUPTI=1`** (uses CUPTI /
`torch.profiler` device time instead of CUDA events). cuTile-rs kernels often
have different host/launch overhead than the reference, which CUDA-event wall
timing over-counts on small (sub-microsecond) kernels; CUPTI measures pure GPU
kernel time and gives a stable, apples-to-apples ratio:

```bash
CUPTI=1 pytest tests/ops/test_bmm.py -k "test_perf and cutile_rs" --print-record
```

## Contributing

We welcome contributions of all kinds. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including the Contributor License Agreement (CLA) process.

## License and third-party notices

- Project license: MIT
  - [LICENSE](LICENSE)
- Third-party attributions and license texts:
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
