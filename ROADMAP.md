<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# 🗺️ TileGym Kernel Roadmap & Contribution Guide

Welcome to the TileGym roadmap! We use this page to provide transparency into our development progress and to invite the community to help us build the next generation of tile-based high-performance kernels.

## 1. Current Support Status

### 1.1 Operator Support

The following table tracks the support status of TileGym's core operators, the
ones under `tilegym.ops`. Operators ported from third-party kernel libraries keep
their own namespace (`tilegym.suites.*`) and are tracked in §1.3 instead.

| Category | Operator | Forward | Backward |
|---|---|---|---|
| Linear Algebra | Batch MatMul (BMM) | ✅ Available | 📅 Planned |
| Linear Algebra | FP8 Quantized MatMul | ✅ Available | N/A |
| Linear Algebra | Grouped GEMM | ✅ Available | N/A |
| Linear Algebra | MatMul | ✅ Available | 📅 Planned |
| Linear Algebra | Split-K Reduction | ✅ Available | N/A |
| Attention | Attention | ✅ Available | 🧪 Experimental |
| Attention | Attention Sink | ✅ Available | N/A |
| Attention | Attention Sink Decode | ✅ Available | N/A |
| Attention | Attention Variants | ✅ Available | N/A |
| Attention | Autoregressive Flash Attention | 🚧 WIP (Internal) | N/A |
| Attention | Flash Decode | ✅ Available | N/A |
| Attention | Flex Attention | 📅 Planned | N/A |
| Attention | Gemma Attention | ✅ Available | N/A |
| Attention | Gemma Attention Decode | ✅ Available | N/A |
| Attention | MLA Decoding | ✅ Available | N/A |
| Attention | MLA Decoding Split KV | ✅ Available | N/A |
| Attention | Multi-Head Compression (MHC) | 🧪 Experimental | N/A |
| Attention | Multi-Latent Attention (MLA) | ✅ Available | N/A |
| Attention | Sliding-Window Attention (SWA) | 🧪 Experimental | N/A |
| Attention | Sparse MLA | 🧪 Experimental | N/A |
| Attention | Variable-Length Attention | ✅ Available | N/A |
| Linear Attention / SSM | Chunked Gated Delta Rule | ✅ Available | N/A |
| Linear Attention / SSM | Mamba-2 | ✅ Available | ✅ Available |
| Linear Attention / SSM | Recurrent Gated Delta Rule | ✅ Available | N/A |
| Normalization | Cache Layer Normalization | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Normalization | Group Normalization | 📅 Planned | 📅 Planned |
| Normalization | Layer Normalization | ✅ Available | 📅 Planned |
| Normalization | Layer Normalization Legacy | ✅ Available | 📅 Planned |
| Normalization | RMS Normalization | ✅ Available | 🧪 Experimental |
| Activation | Dropout | ✅ Available | 📅 Planned |
| Activation | GeGLU | ✅ Available | ✅ Available |
| Activation | GeLU | ✅ Available | ✅ Available |
| Activation | ReLU | ✅ Available | 📅 Planned |
| Activation | SiLU and Mul | ✅ Available | 🧪 Experimental |
| Activation | Softmax | ✅ Available | 🚧 WIP (Internal) |
| Activation | SwiGLU | ✅ Available | 🧪 Experimental |
| Fused Operations | Linear + Activation + Linear | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Fused Operations | Linear + Bias + Activation | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Fused Operations | Linear + Elementwise | 🚧 WIP (Internal) | 📅 Planned |
| Fused Operations | Linear + GLU Activation + Linear | ✅ Available | 📅 Planned |
| Mixture of Experts | MoE | ✅ Available | N/A |
| Mixture of Experts | MoE Activation Gradient Backward | N/A | ✅ Available |
| Mixture of Experts | MoE Align Block | ✅ Available | N/A |
| Positional Encoding | Rotary Position Embedding (RoPE) | ✅ Available | ✅ Available |
| Quantization | NVFP4 Quantize | 🧪 Experimental | N/A |
| Tensor Manipulation | Concatenation | 🚧 WIP (Internal) | N/A |
| Tensor Manipulation | Transpose | ✅ Available | ✅ Available |
| Pointwise | Squares | 📅 Planned | N/A |
| Signal Processing | Fast Fourier Transform (FFT) | 🚧 WIP (Internal) | N/A |
| Convolution | Convolution | 📅 Planned | 📅 Planned |
| Loss Functions | Cross Entropy | 📅 Planned | 📅 Planned |
| Loss Functions | Fused Linear Cross Entropy | 🧪 Experimental | N/A |
| Embedding | BERT Embeddings | 🚧 WIP (Internal) | N/A |
| Optimizer | Fused Adam | 📅 Planned | N/A |

The table above tracks cuTile operators. Triton implementations are also available for: `attention_variant`, `dropout`, `layer_norm_legacy`, `rms_norm`, `rope`.

### 1.2 E2E Model Support

The following table tracks the support status for various models.

| Model | Status | Notes |
|---|---|---|
| DeepSeek-V2-Lite-Chat | ✅ Available | Tested on B200 |
| GPT-OSS-20B | ✅ Available | Tested on B200 |
| Gemma-3-4B-IT | ✅ Available | Tested on B200 |
| LFM2-8B-A1B (MoE) | ✅ Available | Tested on B200 |
| LLaMA-3.1-8B | ✅ Available | Tested on B200 |
| Mistral-7B-Instruct-v0.3 | ✅ Available | Tested on B200 |
| OLMo-3-1025-7B | ✅ Available | Tested on B200 |
| OLMoE-1B-7B | ✅ Available | Tested on B200 |
| Phi-3-mini-4k-instruct | ✅ Available | Tested on B200 |
| Qwen2-7B | ✅ Available | Tested on B200 |
| Qwen3.5 | ✅ Available | Tested on B200 |
| More LLM models | 🙋 Help Wanted |  |

### 1.3 Kernel Library Support

The following table tracks the support status for various kernel libraries.

| Library | Status | Notes |
|---|---|---|
| Liger-Kernel | ✅ Available | 25 cuTile kernels |
| FlashInfer | ✅ Available | 8 cuTile kernels |
| Unsloth | ✅ Available | 8 cuTile kernels |
| FlagGems | 🚧 WIP (Internal) |  |
| Tokamax | 🚧 WIP (Internal) |  |
| Other Libraries | 📅 Planned | We welcome suggestions on which repositories you'd like to see cuTile performance in |

### 1.4 Backend Coverage

TileGym kernels are written against more than one backend. The following table
lists those backends and how many kernels are available in each.

| Backend | Status | Kernels Available | Notes |
|---|---|---|---|
| cuTile (Python) | ✅ Available | 81 | Primary backend; the tile-based Python DSL |
| Triton | ✅ Available | 5 | Secondary backend |
| cuTile.rs (Rust) | ✅ Available | 5 | Rust bindings over the same cuTile kernels |
| TileC++ | ✅ Available | 27 | C++/CUDA kernels exposed through a Python wrapper |

### Status Definitions:

- **✅ Available**: Fully tested, performance optimized, and ready for production use.
- **🧪 Experimental**: Functional and available, but carries the `@experimental_kernel` tag — not yet fully performance-validated. Community feedback welcome.
- **🚧 WIP (Internal)**: Currently being developed by the NVIDIA team. (Internal development is active; we recommend waiting for our PR to avoid conflicts).
- **📅 Planned**: On our radar for future development. We are open to design discussions.
- **🙋 Help Wanted**: We would love to have this, but don't have the bandwidth yet. Community contributions are highly encouraged!
- **N/A**: Not applicable — the operator does not implement this direction. Inference-oriented kernels such as the decode paths have no backward pass.

## 2. Contribution Opportunities

We are actively looking for contributors to help with the following strategic areas:

### 🚀 Kernel Implementations (High Priority)

#### Optimize Existing Kernels

Make existing kernels run faster. Our internal optimization efforts currently focus on B200. If you discover optimizations that can make kernels faster, we welcome your contributions. You can choose to add tuning configs for specific architectures. However, if you make changes to the kernel itself, we will internally test whether your optimizations cause performance regressions on all covered GPUs.

#### Submit New Kernels

We welcome contributions of any new kernels, especially kernels required by new models. Before you start implementing, please check existing kernels in the repository, review our roadmap, and search through open issues to ensure that no one else is already working on the same kernel.

### 🔗 E2E Model Support

**New Model Integration**: Help us support more LLM models (e.g., Mixtral, Llama 4 and beyond).

**Model Optimization**: Performance tuning and optimization for existing model support.


## 3. How to Contribute

For detailed contribution guidelines, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

If you want to contribute a new kernel or claim a Help Wanted task:

1. **Review Existing Code**: Check `tilegym/ops/cutile` (e.g., the GEMM implementation) to understand our DSL and coding standards.

2. **Submit a PR**: Directly open a pull request with your implementation. Your PR description must include:
   - Performance profiling data comparing against baseline implementations (e.g., torch, cuBLAS, flashinfer, or Triton).
   - Unit tests covering various shapes.

**For E2E Model Support**: If your contribution involves end-to-end model support and will take a significant amount of time, please open an issue first to discuss your approach and let us know that you are working on it. This helps us coordinate efforts and avoid duplicate work.

If you meet any problems, please [Open an Issue] to let us know. Your feedback helps us prioritize our internal roadmap!
