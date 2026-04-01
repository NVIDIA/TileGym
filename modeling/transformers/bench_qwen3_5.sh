#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Benchmark script for Qwen3.5-0.8B model
# Compares PyTorch baseline vs TileGym CUTILE backend

set -e

MODEL_ID="Qwen/Qwen3.5-0.8B"
INPUT_FILE="sample_inputs/input_prompt_small.txt"
OUTPUT_LENGTH=50
SUMMARY_FILE="qwen3_5_benchmark_summary.txt"
BATCH_SIZE=1
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/tilegym_bench}"

echo "========================================"
echo "  Qwen3.5-0.8B Performance Benchmark"
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
    --log_dir ${LOG_DIR} \
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
    --log_dir ${LOG_DIR} \
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

echo ""
echo "========================================"
echo "  TileGym Kernel Coverage"
echo "========================================"
python infer.py \
    --model_id ${MODEL_ID} \
    --use_tilegym \
    --use_cutile \
    --use_attn \
    --report_kernel_coverage \
    --log_dir ${LOG_DIR} \
    --sentence_file ${INPUT_FILE} \
    --batch_size ${BATCH_SIZE} \
    --output_length ${OUTPUT_LENGTH}
echo "========================================"
