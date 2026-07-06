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

> **GPU 支持**：TileGym 需要 **CUDA 13.1+** 和 **Blackwell GPU**（如 B200、RTX 5080、RTX 5090）。**NVIDIA Ampere**（如 A100）也受支持，但需要 **CUDA 13.2+**。所有已发布的内核均在两种架构上经过验证。请从 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads) 下载 CUDA。

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

TileGym 使用 [`cuda-tile`](https://github.com/nvidia/cutile-python)（≥ 1.3.0）进行 GPU 内核编程，运行时依赖 `tileiras` 编译器。

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

所有运行时依赖均声明在 [`requirements.txt`](requirements.txt) 中，通过 `pip install tilegym` 和 `pip install .` 都会自动安装。

我们还提供了 Dockerfile，您可以参考 [modeling/transformers/README.md](modeling/transformers/README.md)。

### 后端

TileGym 为以下后端提供内核，每个后端的内核位于 `src/tilegym/ops/` 下各自的目录中：

- **cuTile**（默认）—— [`src/tilegym/ops/cutile`](src/tilegym/ops/cutile)，更多详情请参阅 [cutile-python](https://github.com/nvidia/cutile-python)。
- **CUDA Tile C++** —— [`src/tilegym/ops/tilecpp`](src/tilegym/ops/tilecpp)，更多详情请参阅 [README.tilecpp.md](README.tilecpp.md)。
- **Triton CUDA Tile IR** —— [`src/tilegym/ops/triton`](src/tilegym/ops/triton)，更多详情请参阅 [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR)。

要使用 Triton CUDA Tile IR 后端，请将其 wheel 包安装到一个独立的目录中，并在运行时通过 `ENABLE_TILE=1` 选择该后端。[发布页面](https://github.com/triton-lang/Triton-to-tile-IR/releases) 上提供了适用于 CPython 3.12 和 3.13 的 wheel 包：

```bash
# 安装到一个独立的目录，与默认环境隔离
pip install --target /opt/nvtriton <nvtriton-wheel-for-your-python>.whl

# 在运行时选择 Triton CUDA Tile IR 后端
PYTHONPATH=/opt/nvtriton ENABLE_TILE=1 python your_script.py
```

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

### 5. 启用 cuTile-rs (Rust) 后端 (可选)

部分算子在 [`src/tilegym/ops/cutile_rs`](src/tilegym/ops/cutile_rs) 下额外提供了
**cuTile-rs** 后端——内核用 Rust 基于 [`cutile-rs`](https://github.com/NVlabs/cutile-rs)
编写，并通过 C-ABI 的 `libcutile_kernels.so` 加载。该后端为可选项，且仅在源码安装模式下可用。

**前置要求**（在上述基础安装之外），与 [cuTile-rs](https://github.com/NVlabs/cutile-rs) 保持一致：

- **Rust 1.89+**——`cargo` 和 `rustc` 需在 `PATH` 中：

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup default stable
  ```

- **CUDA toolkit 并包含头文件**——Rust 构建会用 `bindgen` 处理 `cuda.h`。请将
  `CUDA_TOOLKIT_PATH` 指向你的安装目录；若未设置，cuTile-rs 会回退到 `/usr/local/cuda`：

  ```bash
  export CUDA_TOOLKIT_PATH=/usr/local/cuda   # 必须包含 include/cuda.h
  ```

**使用方法。** 后端加载器会在首次使用时延迟构建共享库（`cargo build --release`），因此无需手动构建：

```python
import tilegym
tilegym.set_backend("cutile-rs")

from tilegym.backend.selector import get_available_backends
print(get_available_backends())        # 应包含 "cutile-rs"

from tilegym.ops import bmm             # 与后端无关的导入
# ... bmm(...) 现在会分发到 cuTile-rs 内核
```

**可选环境变量：**

```bash
export CUTILE_RS_AUTOBUILD=0                          # 跳过延迟重建；使用预构建的 .so
export CUTILE_RS_KERNELS_DIR=/abs/path/to/cutile_kernels   # 覆盖 crate 位置
```

> 若 `cargo` 不在 `PATH` 中且没有预构建的 `libcutile_kernels.so`，该后端会报告为不可用，
> cuTile-rs 相关测试会被跳过而非失败。

**cuTile-rs 性能测试。** 测量 cuTile-rs 性能时,建议用 **`CUPTI=1`** 运行 perf 测试
(使用 CUPTI / `torch.profiler` 的 device time,而非 CUDA events)。cuTile-rs 内核与参考
实现的 host/launch 开销通常不同,CUDA-event 墙钟计时在亚微秒级小内核上会高估这部分开销;
CUPTI 测的是纯 GPU 内核时间,给出更稳定、可对比的结果:

```bash
CUPTI=1 pytest tests/ops/test_bmm.py -k "test_perf and cutile_rs" --print-record
```

## 贡献

我们欢迎各种形式的贡献。请阅读我们的 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南，包括贡献者许可协议（CLA）流程。

## 许可证与第三方声明

- 项目许可证：MIT
  - [LICENSE](LICENSE)
- 第三方归属和许可证文本：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
