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

> **GPU サポート**: TileGym には **CUDA 13.1+** と **Blackwell GPU**（例：B200、RTX 5080、RTX 5090）が必要です。**NVIDIA Ampere**（例：A100）も **CUDA 13.2+** でサポートされています。リリース済みのすべてのカーネルは両アーキテクチャで検証済みです。CUDA は [NVIDIA CUDA ダウンロード](https://developer.nvidia.com/cuda-downloads) からダウンロードしてください。

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

TileGym は GPU カーネルプログラミングに [`cuda-tile`](https://github.com/nvidia/cutile-python)（≥ 1.3.0）を使用しており、実行時に `tileiras` コンパイラに依存しています。

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

すべてのランタイム依存関係は [`requirements.txt`](requirements.txt) に宣言されており、`pip install tilegym` と `pip install .` の両方で自動的にインストールされます。

Dockerfile も提供しています。[modeling/transformers/README.md](modeling/transformers/README.md) を参照してください。

### バックエンド

TileGym は以下のバックエンド向けのカーネルを提供しており、それぞれ `src/tilegym/ops/` 配下の個別のディレクトリにあります:

- **cuTile**（デフォルト）—— [`src/tilegym/ops/cutile`](src/tilegym/ops/cutile)、詳細は [cutile-python](https://github.com/nvidia/cutile-python) を参照してください。
- **CUDA Tile C++** —— [`src/tilegym/ops/tilecpp`](src/tilegym/ops/tilecpp)、詳細は [README.tilecpp.md](README.tilecpp.md) を参照してください。
- **Triton CUDA Tile IR** —— [`src/tilegym/ops/triton`](src/tilegym/ops/triton)、詳細は [Triton-to-tile-IR](https://github.com/triton-lang/Triton-to-tile-IR) を参照してください。

Triton CUDA Tile IR バックエンドを使用するには、その wheel を別のディレクトリにインストールし、実行時に `ENABLE_TILE=1` で選択します。CPython 3.12 および 3.13 向けの wheel は [リリースページ](https://github.com/triton-lang/Triton-to-tile-IR/releases) で入手できます:

```bash
# デフォルト環境とは分離した別のディレクトリにインストールします
pip install --target /opt/nvtriton <nvtriton-wheel-for-your-python>.whl

# 実行時に Triton CUDA Tile IR バックエンドを選択します
PYTHONPATH=/opt/nvtriton ENABLE_TILE=1 python your_script.py
```

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

### 5. cuTile-rs (Rust) バックエンドの有効化 (オプション)

一部の演算子は [`src/tilegym/ops/cutile_rs`](src/tilegym/ops/cutile_rs) の下に追加の
**cuTile-rs** バックエンドを提供します。カーネルは [`cutile-rs`](https://github.com/NVlabs/cutile-rs)
を用いて Rust で記述され、C-ABI の `libcutile_kernels.so` を介して読み込まれます。
これはオプションであり、ソースからのインストール時のみ利用できます。

**前提条件**（上記の基本インストールに加えて）、[cuTile-rs](https://github.com/NVlabs/cutile-rs) に準拠：

- **Rust 1.89+** — `cargo` と `rustc` が `PATH` にあること：

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup default stable
  ```

- **CUDA toolkit（ヘッダー付き）** — Rust ビルドは `bindgen` で `cuda.h` を処理します。
  `CUDA_TOOLKIT_PATH` をインストール先に設定してください。未設定の場合、cuTile-rs は
  `/usr/local/cuda` にフォールバックします：

  ```bash
  export CUDA_TOOLKIT_PATH=/usr/local/cuda   # include/cuda.h を含む必要があります
  ```

**使い方。** バックエンドローダーは初回使用時に共有ライブラリを遅延ビルドする（`cargo build --release`）ため、手動ビルドは不要です：

```python
import tilegym
tilegym.set_backend("cutile-rs")

from tilegym.backend.selector import get_available_backends
print(get_available_backends())        # "cutile-rs" が含まれるはずです

from tilegym.ops import bmm             # バックエンド非依存のインポート
# ... bmm(...) が cuTile-rs カーネルにディスパッチされます
```

**オプションの環境変数：**

```bash
export CUTILE_RS_AUTOBUILD=0                          # 遅延リビルドをスキップし、ビルド済み .so を使用
export CUTILE_RS_KERNELS_DIR=/abs/path/to/cutile_kernels   # crate の場所を上書き
```

> `cargo` が `PATH` になく、ビルド済みの `libcutile_kernels.so` も存在しない場合、
> バックエンドは利用不可として報告され、cuTile-rs のテストは失敗せずスキップされます。

**cuTile-rs のベンチマーク。** cuTile-rs の性能を測定する際は、**`CUPTI=1`** を付けて perf
テストを実行してください(CUDA events ではなく CUPTI / `torch.profiler` のデバイス時間を使用)。
cuTile-rs カーネルは参照実装と host/launch オーバーヘッドが異なることが多く、CUDA-event の
実時間計測はサブマイクロ秒の小さなカーネルでこれを過大に数えます。CUPTI は純粋な GPU カーネル
時間を測定し、安定した比較可能な結果を与えます:

```bash
CUPTI=1 pytest tests/ops/test_bmm.py -k "test_perf and cutile_rs" --print-record
```

## コントリビューション

あらゆる種類のコントリビューションを歓迎します。ガイドラインについては、コントリビューターライセンス契約（CLA）プロセスを含む [CONTRIBUTING.md](CONTRIBUTING.md) をお読みください。

## ライセンスおよび第三者に関する通知

- プロジェクトライセンス：MIT
  - [LICENSE](LICENSE)
- 第三者の帰属表示とライセンステキスト：
  - [LICENSES/ATTRIBUTIONS.md](LICENSES/ATTRIBUTIONS.md)
