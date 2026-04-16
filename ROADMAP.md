<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# 🗺️ TileGym Kernel Roadmap & Contribution Guide

Welcome to the TileGym roadmap! We use this page to provide transparency into our development progress and to invite the community to help us build the next generation of tile-based high-performance kernels.

## 1. Current Support Status

### 1.1 Operator Support

The following table tracks the support status for various operators.

| Category | Operator | Forward | Backward |
|----------|----------|--------|----------|
| Linear Algebra | MatMul | ✅ Available | N/A |
| Linear Algebra | Batch MatMul (BMM) | ✅ Available | 📅 Planned |
| Linear Algebra | Grouped GEMM | ✅ Available | N/A |
| Linear Algebra | FP8 Quantized MatMul | 🚧 WIP (Internal) | N/A |
| Linear Algebra | Split-K Reduction | ✅ Available | N/A |
| Attention | Attention | ✅ Available | 🧪 Experimental |
| Attention | Flash Decode | ✅ Available | N/A |
| Attention | Attention Sink Decode | ✅ Available | N/A |
| Attention | Attention Sink | ✅ Available | N/A |
| Attention | Autoregressive Flash Attention | 🚧 WIP (Internal) | N/A |
| Attention | Flex Attention | 📅 Planned | N/A |
| Attention | Multi-Head Compression (MHC) | ✅ Available | N/A |
| Attention | Multi-Latent Attention (MLA) | ✅ Available | N/A |
| Attention | MLA Decoding | ✅ Available | N/A |
| Attention | MLA Decoding Split KV | ✅ Available | N/A |
| Normalization | RMS Normalization | ✅ Available | ✅ Available |
| Normalization | Layer Normalization Legacy | ✅ Available | 📅 Planned |
| Normalization | Cache Layer Normalization | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Normalization | Group Normalization | 📅 Planned | N/A |
| Activation | SiLU and Mul | ✅ Available | 🧪 Experimental |
| Activation | SwiGLU | ✅ Available | 🧪 Experimental |
| Activation | Dropout | ✅ Available | N/A |
| Activation | Softmax | ✅ Available | 🚧 WIP (Internal) |
| Fused Operations | Linear + Activation + Linear | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Fused Operations | Linear + Bias + Activation | 🚧 WIP (Internal) | 🚧 WIP (Internal) |
| Fused Operations | Linear + Elementwise | 🚧 WIP (Internal) | N/A |
| Fused Operations | Linear + GLU Activation + Linear | 🚧 WIP (Internal) | 📅 Planned |
| Mixture of Experts | MoE | ✅ Available | N/A |
| Mixture of Experts | MoE Align Block | ✅ Available | N/A |
| Positional Encoding | Rotary Position Embedding (RoPE) | ✅ Available | ✅ Available |
| Tensor Manipulation | Concatenation | 🚧 WIP (Internal) | N/A |
| Tensor Manipulation | Transpose | 🚧 WIP (Internal) | N/A |
| Signal Processing | Fast Fourier Transform (FFT) | 🚧 WIP (Internal) | N/A |
| Convolution | Convolution | 📅 Planned | 📅 Planned |
| Loss Functions | Cross Entropy | 🧪 Experimental | 📅 Planned |
| Embedding | BERT Embeddings | 🚧 WIP (Internal) | N/A |
| Optimizer | Fused Adam | 📅 Planned | N/A |
| Pointwise | Squares | 📅 Planned | N/A |

### 1.2 E2E Model Support

The following table tracks the support status for various models.

| Model | Status | Notes |
|-------|--------|-------|
| LLaMA-3.1-8B | ✅ Available | Tested on B200 |
| DeepSeek-V2-Lite-Chat | ✅ Available | Tested on B200 |
| Qwen2-7B | ✅ Available | Tested on B200 |
| Qwen3.5-7B | ✅ Available | Tested on B200 |
| Gemma-3-4B-IT | ✅ Available | Tested on B200 |
| GPT-OSS | ✅ Available | Tested on B200 |
| Mistral-7B-Instruct-v0.3 | ✅ Available | Tested on B200 |
| Phi-3-mini-4k-instruct | ✅ Available | Tested on B200 |
| OLMo-3-1025-7B | ✅ Available | Tested on B200 |
| More LLM models | 🙋 Help Wanted | |

### 1.3 Kernel Library Support

The following table tracks the support status for various kernel libraries.

| Library | Status | Notes |
|---------|--------|-------|
| Flashinfer | 🚧 WIP (Internal) | |
| Tokamax | 🚧 WIP (Internal) | |
| Flaggems | 🚧 WIP (Internal) |  |
| Other Libraries | 📅 Planned | We welcome suggestions on which repositories you'd like to see cuTile performance in |

### Status Definitions:

- **✅ Available**: Fully tested, performance optimized, and ready for production use.
- **🧪 Experimental**: Functional and available, but carries the `@experimental_kernel` tag — not yet fully performance-validated. Community feedback welcome.
- **🚧 WIP (Internal)**: Currently being developed by the NVIDIA team. (Internal development is active; we recommend waiting for our PR to avoid conflicts).
- **📅 Planned**: On our radar for future development. We are open to design discussions.
- **🙋 Help Wanted**: We would love to have this, but don't have the bandwidth yet. Community contributions are highly encouraged!

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
