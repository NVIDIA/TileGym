<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# TileGym Transformers Inference

End-to-end inference examples for transformer language models accelerated with TileGym kernels. Optimized for NVIDIA Blackwell Architecture.

## Supported Models

| Model | Model ID | Features |
|-------|----------|----------|
| LLaMA-3.1-8B | `meta-llama/Meta-Llama-3.1-8B` | RoPE, SwiGLU, RMSNorm, Attention*, Flash Decoding* |
| DeepSeek-V2-Lite-Chat | `deepseek-ai/DeepSeek-V2-Lite-Chat` | RoPE, SwiGLU, RMSNorm, MoE, MLADecoding*, Attention* |
| Qwen2-7B | `Qwen/Qwen2-7B` | RoPE, SwiGLU, RMSNorm, Attention* |

*Optional: Enable with `--use_attn`, we can use attention provided in TileGym

B200 can support both models. Due to memory constraints, RTX 5090 GPUs only support LLaMA-3.1-8B models. DeepSeek-V2-Lite-Chat requires higher memory capacity.

## Docker Support

```bash
# Option 1: Use the build script
cd modeling/transformers
./build_docker.sh

# Option 2: Build manually (must run from tilegym repository root)
cd /path/to/tilegym
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .

# Enter interactive mode
docker run --gpus all -it tilegym-transformers bash

# Or run inference directly
docker run --gpus all -it tilegym-transformers \
    python infer.py --model_id deepseek-ai/DeepSeek-V2-Lite-Chat --use_tilegym --use_cutile --use_attn --show_outputs
```

## Quick Start

### Basic Inference

```bash
# Transformer baseline
python infer.py --model_id meta-llama/Meta-Llama-3.1-8B --show_outputs

# With CUTILE backend
python infer.py --model_id meta-llama/Meta-Llama-3.1-8B --use_tilegym --use_cutile --use_attn --show_outputs
```

### Using Custom Inputs

```bash
# From file
python infer.py \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym \
    --use_attn \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --output_length 100

# From command line
python infer.py \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --input_text "Explain machine learning" \
    --show_outputs
```

### Performance Profiling

Will provide results using Torch Profiler.
```bash
python infer.py \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --num_runs 5
```

## Performance Benchmark

Benchmark TileGym's CUTILE-optimized kernels against standard PyTorch implementation. The `--profile` flag enables detailed performance metrics including throughput (tokens/sec) and generation latency.

### Quick Start

Run benchmark scripts for automated comparison:

```bash
# LLaMA-3.1-8B benchmark
./bench_llama.sh

# DeepSeek-V2-Lite benchmark
./bench_deepseek.sh

# Qwen2-7B benchmark
./bench_qwen.sh
```

### Manual Benchmark

#### LLaMA-3.1-8B Benchmark

```bash
# PyTorch baseline
python infer.py \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --profile \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --output_length 100

# TileGym CUTILE backend
python infer.py \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --output_length 100
```

#### DeepSeek-V2-Lite Benchmark

```bash
# PyTorch baseline
python infer.py \
    --model_id deepseek-ai/DeepSeek-V2-Lite-Chat \
    --profile \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --output_length 100

# TileGym CUTILE backend
python infer.py \
    --model_id deepseek-ai/DeepSeek-V2-Lite-Chat \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --output_length 100
```

#### Qwen2-7B Benchmark
```bash
# PyTorch baseline
python infer.py \
    --model_id Qwen/Qwen2-7B \
    --profile \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --batch_size 16 \
    --output_length 100

# TileGym CUTILE backend
python infer.py \
    --model_id Qwen/Qwen2-7B \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --batch_size 16 \
    --output_length 100
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_id` | HuggingFace model ID or local path | `meta-llama/Meta-Llama-3.1-8B` |
| `--use_tilegym` | Enable TileGym optimization | `False` |
| `--use_cutile` | Use CUTILE backend | `False` |
| `--use_attn` | Enable attention optimization | `False` |
| `--input_text` | Input prompt text | - |
| `--sentence_file` | Input file path | - |
| `--output_length` | Number of tokens to generate | `100` |
| `--batch_size` | Batch size | `1` |
| `--precision` | `bfloat16` or `float32` | `bfloat16` |
| `--num_runs` | Benchmark iterations | `5` |
| `--warmup_runs` | Warmup iterations | `2` |
| `--profile` | Enable profiling | `False` |
| `--show_outputs` | Print generated text | `False` |


## Using Local Models

You can use locally cached models by specifying the path directly:

```bash
# Use local model path
python infer.py \
    --model_id /path/to/local/model \
    --use_tilegym \
    --use_attn \
    --use_cutile \
    --show_outputs
```

## Troubleshooting

**CUDA Out of Memory**
- Reduce `--batch_size` or use smaller inputs
- Use `--precision bfloat16`

**Model Download Issues**
- Set up [HuggingFace authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- Ensure your account has access to the model (e.g., [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B))
- Check network connectivity

**Slow Performance**
- Disable `--use_tilegym` flag to see whether naive version has output
- Sometimes, it may take more than one minute to get the output when your output sentence is too long. Try to use shorter input and reduce `--output_length`

**Import Errors**
- Install TileGym: `pip install -e .` (from repo root)
