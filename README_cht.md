<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

[English](README.md) | [简体中文](README_chs.md) | 繁體中文 | [日本語](README_ja.md) | [Français](README_fr.md)

# TileGym

TileGym 是一個 CUDA Tile 核心函式庫，提供了豐富的基於 Tile 的 GPU 程式設計核心教學與範例集合。

[**概述**](#概述) |
[**功能特性**](#功能特性) |
[**安裝**](#安裝) |
[**快速開始**](#快速開始) |
[**貢獻**](#貢獻) |
[**授權條款**](#授權條款與第三方聲明)

## 概述

本儲存庫旨在為基於 Tile 的 GPU 程式設計提供實用的核心教學與範例。TileGym 是一個用於體驗 CUDA Tile 的實驗平台，您可以在此學習如何建構高效的 GPU 核心，並探索它們在 Llama 3.1 和 DeepSeek V2 等實際大型語言模型中的整合應用。無論您是正在學習基於 Tile 的 GPU 程式設計，還是希望最佳化您的大型語言模型實作，TileGym 都能提供實用的範例和全面的指導。
<img width="95%" alt="tilegym_1_newyear" src="https://github.com/user-attachments/assets/f37010f5-14bc-44cd-bddf-f517dc9922b8" />

## 功能特性

- 豐富的 CUDA Tile 核心範例集合
- 常見深度學習運算子的實用核心實作
- 用於評估核心效率的效能基準測試
- 與主流大型語言模型（Llama 3.1、DeepSeek V2）的端到端整合範例

## 安裝

### 前置需求

> ⚠️ **重要提示**：TileGym 需要 **CUDA 13.1+** 和 **NVIDIA Blackwell 架構 GPU**（如 B200、RTX 5080、RTX 5090）。我們將在未來支援其他 GPU 架構。請從 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads) 下載 CUDA。

- PyTorch（版本 2.9.1 或相容版本）
- **[CUDA 13.1+](https://developer.nvidia.com/cuda-downloads)**（必需 - TileGym 僅在 CUDA 13.1+ 上建構和測試）
- Triton（隨 PyTorch 安裝一起包含）

### 安裝步驟

#### 1. 準備 `torch` 和 `triton` 環境

如果您已經安裝了 `torch` 和 `triton`，請跳過此步驟。

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

我們已驗證 `torch==2.9.1` 可以正常運作。安裝 `torch` 時也會自動取得 `triton` 套件。

#### 2. 安裝 TileGym

TileGym 使用 [`cuda-tile`](https://github.com/nvidia/cutile-python) 進行 GPU 核心程式設計，執行時期依賴 `tileiras` 編譯器。

##### 從 PyPI 安裝（建議）

```bash
pip install tilegym[tileiras]
```

這將安裝 TileGym 及其所有執行時期依賴，包括 `cuda-tile[tileiras]`，它會將 `tileiras` 編譯器直接捆綁到您的 Python 環境中。

如果您的系統上已有 `tileiras`（例如來自 [CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads)），可以省略附加選項：

```bash
pip install tilegym
```

##### 從原始碼安裝

```bash
git clone https://github.com/NVIDIA/TileGym.git
cd TileGym
pip install .[tileiras]   # 或者: pip install .  (如果您已有系統級 tileiras)
```

如需可編輯（開發）模式，請使用 `pip install -e .` 或 `pip install -e .[tileiras]`。

##### 安裝 `cuda-tile-experimental`

> ⚠️ **必需**：TileGym 核心使用了 [`cuda-tile-experimental`](https://github.com/NVIDIA/cutile-python/tree/main/experimental) 中的功能（如自動調優器）。此套件*不*在 PyPI 上提供，必須從原始碼單獨安裝：
>
> ```bash
> pip install "cuda-tile-experimental @ git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"
> ```
>
> `cuda-tile-experimental` 由 CUDA Tile 團隊維護，僅提供原始碼安裝。更多詳情請參閱 [experimental-features-optional](https://github.com/NVIDIA/cutile-python?tab=readme-ov-file#experimental-features-optional)。

所有執行時期依賴（`cuda-tile-experimental` 除外）均宣告於 [`requirements.txt`](requirements.txt) 中，透過 `pip install tilegym` 和 `pip install .` 都會自動安裝。

我們還提供了 Dockerfile，您可以參考 [modeling/transformers/README.md](modeling/transformers/README.md)。

## 快速開始

TileGym 有三種主要使用方式：

### 1. 探索核心範例

所有核心實作位於 `src/tilegym/ops/` 目錄下。您可以使用簡潔的腳本測試單一操作。函式級用法和單一運算子的最小腳本文件詳見 [tests/ops/README.md](tests/ops/README.md)

### 2. 執行基準測試

使用微基準測試評估核心效能：

```bash
cd tests/benchmark
bash run_all.sh
```

完整的基準測試指南詳見 [tests/benchmark/README.md](tests/benchmark/README.md)

### 3. 執行 LLM Transformer 範例

在端到端推理場景中使用 TileGym 核心。我們提供了可執行的腳本和說明，用於使用 TileGym 核心加速的 Transformer 語言模型（如 Llama 3.1-8B）。

首先，安裝額外依賴：

```bash
pip install accelerate==1.13.0 --no-deps
```

**容器化部署（Docker）**：

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

更多詳情請參閱 [modeling/transformers/README.md](modeling/transformers/README.md)

### 4. Julia (cuTile.jl) 核心 (選填)

TileGym 還包含在 Julia 中實作的實驗性 [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) 核心。這些核心獨立存在於 `julia/` 目錄中，不需要安裝 Python 版 TileGym 套件。

**前置需求**：[Julia 1.12+](https://julialang.org/downloads/)、CUDA 13.1、Blackwell 架構 GPU

```bash
# 安裝 Julia（若尚未安裝）
curl -fsSL https://install.julialang.org | sh

# 安裝依賴
julia --project=julia/ -e 'using Pkg; Pkg.instantiate()'

# 執行測試
julia --project=julia/ julia/test/runtests.jl
```

完整依賴列表請參閱 `julia/Project.toml`。

## 貢獻

我們歡迎各種形式的貢獻。請閱讀我們的 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南，包括貢獻者授權協議（CLA）流程。

## 授權條款與第三方聲明

- 專案授權條款：MIT
  - [LICENSE](LICENSE)
- 第三方歸屬和授權條款文本：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
