<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

[English](README.md) | [简体中文](README_chs.md) | [繁體中文](README_cht.md) | 日本語 | [Français](README_fr.md)

# TileGym

TileGym は、タイルベースの GPU プログラミングのための豊富なカーネルチュートリアルとサンプルを提供する CUDA Tile カーネルライブラリです。

[**概要**](#概要) |
[**機能**](#機能) |
[**インストール**](#インストール) |
[**クイックスタート**](#クイックスタート) |
[**コントリビューション**](#コントリビューション) |
[**ライセンス**](#ライセンスおよび第三者に関する通知)

## 概要

このリポジトリは、タイルベースの GPU プログラミングに役立つカーネルチュートリアルとサンプルを提供することを目的としています。TileGym は CUDA Tile を体験するためのプレイグラウンドであり、効率的な GPU カーネルの構築方法を学び、Llama 3.1 や DeepSeek V2 などの実際の大規模言語モデルへの統合を探索できます。タイルベースの GPU プログラミングの学習中の方も、LLM 実装の最適化を目指している方も、TileGym は実践的なサンプルと包括的なガイダンスを提供します。
<img width="95%" alt="tilegym_1_newyear" src="https://github.com/user-attachments/assets/f37010f5-14bc-44cd-bddf-f517dc9922b8" />

## 機能

- 豊富な CUDA Tile カーネルサンプル集
- 一般的なディープラーニング演算子の実用的なカーネル実装
- カーネル効率を評価するためのパフォーマンスベンチマーク
- 人気のある LLM（Llama 3.1、DeepSeek V2）とのエンドツーエンド統合サンプル

## インストール

### 前提条件

> **GPU サポート**: TileGym には **CUDA 13.1+** と **NVIDIA Ampere**（例：A100）または **Blackwell GPU**（例：B200、RTX 5080、RTX 5090）が必要です。リリース済みのすべての cuTile カーネルは両アーキテクチャで検証済みです。Ampere のパフォーマンスは現在も積極的に最適化中です。CUDA は [NVIDIA CUDA ダウンロード](https://developer.nvidia.com/cuda-downloads) からダウンロードしてください。

- PyTorch（バージョン 2.9.1 または互換バージョン）
- **[CUDA 13.1+](https://developer.nvidia.com/cuda-downloads)**（必須 - TileGym は CUDA 13.1+ でのみビルドおよびテストされています）
- Triton（PyTorch のインストールに含まれます）

### セットアップ手順

#### 1. `torch` と `triton` 環境の準備

すでに `torch` と `triton` がインストールされている場合は、この手順をスキップしてください。

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

`torch==2.9.1` で動作確認済みです。`torch` をインストールする際に `triton` パッケージも自動的に取得されます。

#### 2. TileGym のインストール

TileGym は GPU カーネルプログラミングに [`cuda-tile`](https://github.com/nvidia/cutile-python) を使用しており、実行時に `tileiras` コンパイラに依存しています。

##### PyPI からインストール（推奨）

```bash
pip install tilegym[tileiras]
```

これにより、TileGym とすべてのランタイム依存関係がインストールされます。`cuda-tile[tileiras]` が含まれており、`tileiras` コンパイラが Python 環境に直接バンドルされます。

システムに `tileiras` が既にインストールされている場合（例：[CUDA Toolkit 13.1+](https://developer.nvidia.com/cuda-downloads) から）、追加オプションを省略できます：

```bash
pip install tilegym
```

##### ソースからインストール

```bash
git clone https://github.com/NVIDIA/TileGym.git
cd TileGym
pip install .[tileiras]   # または: pip install .  (システムに tileiras がある場合)
```

編集可能（開発）モードの場合は、`pip install -e .` または `pip install -e .[tileiras]` を使用してください。

##### `cuda-tile-experimental` のインストール

> ⚠️ **必須**：TileGym カーネルは [`cuda-tile-experimental`](https://github.com/NVIDIA/cutile-python/tree/main/experimental) の機能（例：オートチューナー）を使用しています。このパッケージは PyPI では提供されて*おらず*、ソースから個別にインストールする必要があります：
>
> ```bash
> pip install "cuda-tile-experimental @ git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"
> ```
>
> `cuda-tile-experimental` は CUDA Tile チームによってソースのみの実験的パッケージとして管理されています。詳細は [experimental-features-optional](https://github.com/NVIDIA/cutile-python?tab=readme-ov-file#experimental-features-optional) をご覧ください。

すべてのランタイム依存関係（`cuda-tile-experimental` を除く）は [`requirements.txt`](requirements.txt) に宣言されており、`pip install tilegym` と `pip install .` の両方で自動的にインストールされます。

Dockerfile も提供しています。[modeling/transformers/README.md](modeling/transformers/README.md) を参照してください。

## クイックスタート

TileGym には主に3つの使用方法があります：

### 1. カーネルサンプルの探索

すべてのカーネル実装は `src/tilegym/ops/` ディレクトリにあります。最小限のスクリプトで個々の操作をテストできます。関数レベルの使用方法と個々の演算子の最小スクリプトは [tests/ops/README.md](tests/ops/README.md) に記載されています。

### 2. ベンチマークの実行

マイクロベンチマークでカーネルパフォーマンスを評価：

```bash
cd tests/benchmark
bash run_all.sh
```

完全なベンチマークガイドは [tests/benchmark/README.md](tests/benchmark/README.md) で確認できます。

### 3. LLM Transformer サンプルの実行

エンドツーエンドの推論シナリオで TileGym カーネルを使用します。TileGym カーネルで高速化された Transformer 言語モデル（例：Llama 3.1-8B）の実行可能なスクリプトと手順を提供しています。

まず、追加の依存関係をインストールします：

```bash
pip install accelerate==1.13.0 --no-deps
```

**コンテナ化セットアップ（Docker）**：

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

詳細は [modeling/transformers/README.md](modeling/transformers/README.md) をご覧ください。

### 4. Julia (cuTile.jl) カーネル (オプション)

TileGym には、Julia による実験的な [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) カーネル実装も含まれています。これらは `julia/` ディレクトリに独立して収められており、Python の TileGym パッケージを必要としません。

**前提条件**: [Julia 1.12+](https://julialang.org/downloads/)、CUDA 13.1、Blackwell GPU

```bash
# Julia のインストール（未インストールの場合）
curl -fsSL https://install.julialang.org | sh

# 依存関係のインストール
julia --project=julia/ -e 'using Pkg; Pkg.instantiate()'

# テストの実行
julia --project=julia/ julia/test/runtests.jl
```

依存関係の詳細は `julia/Project.toml` を参照してください。

## コントリビューション

あらゆる種類のコントリビューションを歓迎します。ガイドラインについては、コントリビューターライセンス契約（CLA）プロセスを含む [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。

## ライセンスおよび第三者に関する通知

- プロジェクトライセンス：MIT
  - [LICENSE](LICENSE)
- 第三者の帰属表示とライセンステキスト：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
