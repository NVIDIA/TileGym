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

> ⚠️ **重要**: TileGym には **CUDA 13.1** と **NVIDIA Blackwell アーキテクチャ GPU**（例：B200、RTX 5080、RTX 5090）が必要です。今後、他の GPU アーキテクチャもサポートする予定です。CUDA は [NVIDIA CUDA ダウンロード](https://developer.nvidia.com/cuda-downloads) からダウンロードしてください。

- PyTorch（バージョン 2.9.1 または互換バージョン）
- **[CUDA 13.1](https://developer.nvidia.com/cuda-downloads)**（必須 - TileGym は CUDA 13.1 でのみビルドおよびテストされています）
- Triton（PyTorch のインストールに含まれます）

### セットアップ手順

#### 1. `torch` と `triton` 環境の準備

すでに `torch` と `triton` がインストールされている場合は、この手順をスキップしてください。

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/cu130
```

`torch==2.9.1` で動作確認済みです。`torch` をインストールする際に `triton` パッケージも自動的に取得されます。

#### 2. TileGym のインストール

```bash
git clone <tilegym-repository-url>
cd tilegym
pip install .
```
`cuda-tile` が自動的にインストールされます。詳細は https://github.com/nvidia/cutile-python をご覧ください。

`TileGym` を編集モードで使用する場合は、`pip install -e .` を実行してください。

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
pip install accelerate --no-deps
```

**コンテナ化セットアップ（Docker）**：

```bash
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

詳細は [modeling/transformers/README.md](modeling/transformers/README.md) をご覧ください。

## コントリビューション

あらゆる種類のコントリビューションを歓迎します。ガイドラインについては、コントリビューターライセンス契約（CLA）プロセスを含む [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。

## ライセンスおよび第三者に関する通知

- プロジェクトライセンス：MIT
  - [LICENSE](LICENSE)
- 第三者の帰属表示とライセンステキスト：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
