#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Benchmark script for Qwen2-7B model
# Compares PyTorch baseline vs TileGym CUTILE backend

set -e

MODEL_ID="Qwen/Qwen2-7B"
INPUT_FILE="sample_inputs/input_prompt_small.txt"
OUTPUT_LENGTH=50
SUMMARY_FILE="qwen_benchmark_summary.txt"
BATCH_SIZE=16

echo "========================================"
echo "  Qwen2-7B Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_ID}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Clean previous results
rm -f ${SUMMARY_FILE}

echo "Running PyTorch baseline..."
python infer.py \
    --model_id ${MODEL_ID} \
    --profile \
    --sentence_file ${INPUT_FILE} \
    --batch_size ${BATCH_SIZE} \
    --output_length ${OUTPUT_LENGTH} \
    --summary_file ${SUMMARY_FILE}

echo ""
echo "Running TileGym CUTILE backend..."
python infer.py \
    --model_id ${MODEL_ID} \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --profile \
    --sentence_file ${INPUT_FILE} \
    --batch_size ${BATCH_SIZE} \
    --output_length ${OUTPUT_LENGTH} \
    --summary_file ${SUMMARY_FILE}

echo ""
echo "========================================"
echo "  Benchmark Results"
echo "========================================"
if [ -f ${SUMMARY_FILE} ]; then
    cat ${SUMMARY_FILE}
    rm -f ${SUMMARY_FILE}
else
    echo "Summary file not found."
fi
echo "========================================"
