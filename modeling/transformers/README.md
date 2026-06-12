<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# TileGym HF Bench

`modeling/transformers` is a uv-compatible subproject for Hugging Face inference benchmarks, PyTorch profiler traces, and nsys cuTile kernel coverage reports.

The distribution name is `tilegym-hf-bench`; the Python package is `tilegym_hf_bench`.

## Setup

```bash
cd modeling/transformers
uv sync --locked
```

The legacy command remains available:

```bash
python infer.py --help
```

The preferred CLI is:

```bash
uv run tilegym-hf-bench --help
```

## Basic Inference

```bash
uv run tilegym-hf-bench \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --show_outputs

uv run tilegym-hf-bench \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --show_outputs
```

## Profiling

```bash
uv run tilegym-hf-bench \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --num_runs 5
```

## Kernel Coverage

`--report_kernel_coverage` runs the model under `nsys profile` and reports the fraction of GPU time and launches attributed to TileGym/cuTile kernels.

```bash
uv run tilegym-hf-bench \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --report_kernel_coverage \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --output_length 100
```

Kernel-name matching is configured in `src/tilegym_hf_bench/kernel_filters/tilegym_kernel_prefixes.yaml`.

## Benchmark Script

Use one consolidated script for model presets:

```bash
./scripts/benchmark_hf_model.sh --model-key llama
./scripts/benchmark_hf_model.sh --model-key deepseek
./scripts/benchmark_hf_model.sh --model-key qwen
./scripts/benchmark_hf_model.sh --model-key qwen3_5
./scripts/benchmark_hf_model.sh --model-key gemma3
./scripts/benchmark_hf_model.sh --model-key gpt_oss
./scripts/benchmark_hf_model.sh --model-key mistral
./scripts/benchmark_hf_model.sh --model-key phi3
```

Useful overrides:

```bash
./scripts/benchmark_hf_model.sh \
    --model-key qwen3_5 \
    --output-length 16 \
    --batch-size 1 \
    --log-dir /logs/qwen3_5
```

## Docker

```bash
cd /path/to/tilegym
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .
docker run --gpus all -it tilegym-transformers bash
```

The Docker image installs this subproject through uv and keeps `tilegym-hf-bench` on `PATH`.
