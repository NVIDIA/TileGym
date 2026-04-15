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

> **GPU Support**: TileGym requires **CUDA 13.1+** and an **NVIDIA Ampere** (e.g., A100) or **Blackwell GPU** (e.g., B200, RTX 5080, RTX 5090). All released cuTile kernels are validated on both architectures. Note that Ampere performance is still being actively optimized. Download CUDA from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

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

TileGym uses [`cuda-tile`](https://github.com/nvidia/cutile-python) for GPU kernel programming, which depends on the `tileiras` compiler at runtime.

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

##### Install `cuda-tile-experimental`

> ⚠️ **Required**: TileGym kernels use features from [`cuda-tile-experimental`](https://github.com/NVIDIA/cutile-python/tree/main/experimental) (e.g., the autotuner). This package is *not* available on PyPI and must be installed separately from source:
>
> ```bash
> pip install "cuda-tile-experimental @ git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"
> ```
>
> `cuda-tile-experimental` is maintained by the CUDA Tile team as a source-only experimental package. See more details in [experimental-features-optional](https://github.com/NVIDIA/cutile-python?tab=readme-ov-file#experimental-features-optional).

All runtime dependencies (except `cuda-tile-experimental`) are declared in [`requirements.txt`](requirements.txt) and are installed automatically by both `pip install tilegym` and `pip install .`.

We also provide Dockerfile, you can refer to [modeling/transformers/README.md](modeling/transformers/README.md).

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

## Contributing

We welcome contributions of all kinds. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including the Contributor License Agreement (CLA) process.

## License and third-party notices

- Project license: MIT
  - [LICENSE](LICENSE)
- Third-party attributions and license texts:
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
