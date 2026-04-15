<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

[English](README.md) | 简体中文 | [繁體中文](README_cht.md) | [日本語](README_ja.md) | [Français](README_fr.md)

# TileGym

TileGym 是一个 CUDA Tile 内核库，提供了丰富的基于 Tile 的 GPU 编程内核教程和示例集合。

[**概述**](#概述) |
[**功能特性**](#功能特性) |
[**安装**](#安装) |
[**快速开始**](#快速开始) |
[**贡献**](#贡献) |
[**许可证**](#许可证与第三方声明)

## 概述

本仓库旨在为基于 Tile 的 GPU 编程提供有用的内核教程和示例。TileGym 是一个用于体验 CUDA Tile 的实验平台，您可以在这里学习如何构建高效的 GPU 内核，并探索它们在 Llama 3.1 和 DeepSeek V2 等实际大语言模型中的集成应用。无论您是正在学习基于 Tile 的 GPU 编程，还是希望优化您的大语言模型实现，TileGym 都能提供实用的示例和全面的指导。
<img width="95%" alt="tilegym_1_newyear" src="https://github.com/user-attachments/assets/f37010f5-14bc-44cd-bddf-f517dc9922b8" />

## 功能特性

- 丰富的 CUDA Tile 内核示例集合
- 常见深度学习算子的实用内核实现
- 用于评估内核效率的性能基准测试
- 与主流大语言模型（Llama 3.1、DeepSeek V2）的端到端集成示例

## 安装

### 前置要求

> **GPU 支持**：TileGym 需要 **CUDA 13.1+** 和 **NVIDIA Ampere**（如 A100）或 **Blackwell GPU**（如 B200、RTX 5080、RTX 5090）。所有已发布的 cuTile 内核均在两种架构上经过验证。注意，Ampere 性能仍在积极优化中。请从 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads) 下载 CUDA。

- PyTorch（版本 2.9.1 或兼容版本）
- **[CUDA 13.1+](https://developer.nvidia.com/cuda-downloads)**（必需 - TileGym 仅在 CUDA 13.1+ 上构建和测试）
- Triton（随 PyTorch 安装一起包含）

### 安装步骤

#### 1. 准备 `torch` 和 `triton` 环境

如果您已经安装了 `torch` 和 `triton`，请跳过此步骤。

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

我们已验证 `torch==2.9.1` 可以正常工作。安装 `torch` 时也会自动获取 `triton` 包。

#### 2. 安装 TileGym

TileGym 使用 [`cuda-tile`](https://github.com/nvidia/cutile-python) 进行 GPU 内核编程，运行时依赖 `tileiras` 编译器。

##### 从 PyPI 安装（推荐）

```bash
pip install tilegym[tileiras]
```

这将安装 TileGym 及其所有运行时依赖，包括 `cuda-tile[tileiras]`，它会将 `tileiras` 编译器直接捆绑到您的 Python 环境中。

如果您的系统上已有 `tileiras`（例如来自 [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads)），可以省略附加选项：

```bash
pip install tilegym
```

##### 从源码安装

```bash
git clone https://github.com/NVIDIA/TileGym.git
cd TileGym
pip install .[tileiras]   # 或者: pip install .  (如果您已有系统级 tileiras)
```

如需可编辑（开发）模式，请使用 `pip install -e .` 或 `pip install -e .[tileiras]`。

##### 安装 `cuda-tile-experimental`

> ⚠️ **必需**：TileGym 内核使用了 [`cuda-tile-experimental`](https://github.com/NVIDIA/cutile-python/tree/main/experimental) 中的功能（如自动调优器）。此包*不*在 PyPI 上提供，必须从源码单独安装：
>
> ```bash
> pip install "cuda-tile-experimental @ git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"
> ```
>
> `cuda-tile-experimental` 由 CUDA Tile 团队维护，仅提供源码安装。更多详情请参阅 [experimental-features-optional](https://github.com/NVIDIA/cutile-python?tab=readme-ov-file#experimental-features-optional)。

所有运行时依赖（`cuda-tile-experimental` 除外）均声明在 [`requirements.txt`](requirements.txt) 中，通过 `pip install tilegym` 和 `pip install .` 都会自动安装。

我们还提供了 Dockerfile，您可以参考 [modeling/transformers/README.md](modeling/transformers/README.md)。

## 快速开始

TileGym 有三种主要使用方式：

### 1. 探索内核示例

所有内核实现位于 `src/tilegym/ops/` 目录下。您可以使用简洁的脚本测试单个操作。函数级用法和单个算子的最小脚本文档详见 [tests/ops/README.md](tests/ops/README.md)

### 2. 运行基准测试

使用微基准测试评估内核性能：

```bash
cd tests/benchmark
bash run_all.sh
```

完整的基准测试指南详见 [tests/benchmark/README.md](tests/benchmark/README.md)

### 3. 运行 LLM Transformer 示例

在端到端推理场景中使用 TileGym 内核。我们提供了可运行的脚本和说明，用于使用 TileGym 内核加速的 Transformer 语言模型（如 Llama 3.1-8B）。

首先，安装额外依赖：

```bash
pip install accelerate==1.13.0 --no-deps
```

**容器化部署（Docker）**：

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

更多详情请参阅 [modeling/transformers/README.md](modeling/transformers/README.md)

### 4. Julia (cuTile.jl) 内核 (可选)

TileGym 还包含在 Julia 中实现的实验性 [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) 内核。这些内核独立存在于 `julia/` 目录中，不需要安装 Python 版 TileGym 包。

**前置要求**：[Julia 1.12+](https://julialang.org/downloads/)、CUDA 13.1、Blackwell 架构 GPU

```bash
# 安装 Julia（若尚未安装）
curl -fsSL https://install.julialang.org | sh

# 安装依赖
julia --project=julia/ -e 'using Pkg; Pkg.instantiate()'

# 运行测试
julia --project=julia/ julia/test/runtests.jl
```

完整依赖列表请参阅 `julia/Project.toml`。

## 贡献

我们欢迎各种形式的贡献。请阅读我们的 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南，包括贡献者许可协议（CLA）流程。

## 许可证与第三方声明

- 项目许可证：MIT
  - [LICENSE](LICENSE)
- 第三方归属和许可证文本：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
