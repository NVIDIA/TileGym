<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

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
<img width="90%" alt="TileGym_repo" src="https://github.com/user-attachments/assets/1d8741f0-f15c-49ff-ad5c-32d1ae6ec71e" />

## Features

- Rich collection of CUDA Tile kernel examples
- Practical kernel implementations for common deep learning operators
- Performance benchmarking to evaluate kernel efficiency
- End-to-end integration examples with popular LLMs (Llama 3.1, DeepSeek V2)

## Installation

### Prerequisites

> ⚠️ **Important**: TileGym requires **CUDA 13.1** and **NVIDIA Blackwell architecture GPUs** (e.g., B200, RTX 5080, RTX 5090). We will support other GPU architectures in the future. Download CUDA from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

- PyTorch (version 2.9.1 or compatible)
- **[CUDA 13.1](https://developer.nvidia.com/cuda-downloads)** (Required - TileGym is built and tested exclusively on CUDA 13.1)
- Triton (included with PyTorch installation)

### Setup Steps

#### 1. Prepare `torch` and `triton` environment

If you already have `torch` and `triton`, skip this step.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

We have verified that `torch==2.9.1` works. You can also get `triton` packages when installing `torch`.

#### 2. Install TileGym

```bash
git clone <tilegym-repository-url>
cd tilegym
pip install .
```
It will automatically install `cuda-tile`, see https://github.com/nvidia/cutile-python.

If you want to use edit mode for `TileGym`, run `pip install -e .`

We also provide Dockfile, you can refer to [modeling/transformers/README.md](modeling/transformers/README.md).

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
pip install accelerate --no-deps
```

**Containerized Setup (Docker)**:

If you would like to build the TileGym docker image locally, you can use the following commands to do so:

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

If you would like to use pre-built docker, you can run the following command:

```bash
docker run --gpus all -it ghcr.io/nvidia/tilegym:latest bash
```
*Note: We recommend using the "latest" tag, as it represents the most recent tested image.*

More details in [modeling/transformers/README.md](modeling/transformers/README.md)

## Contributing

We welcome contributions of all kinds. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including the Contributor License Agreement (CLA) process.

## License and third-party notices

- Project license: MIT
  - [LICENSE](LICENSE)
- Third-party attributions and license texts:
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
