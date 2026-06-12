#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_BASELINE=1
RUN_CUTILE=1
RUN_COVERAGE=1
MODEL_KEY=""
MODEL_ID=""
INPUT_FILE=""
OUTPUT_LENGTH=""
BATCH_SIZE=""
SUMMARY_FILE=""
LOG_DIR="${LOG_DIR:-/logs}"

usage() {
    cat <<EOF
Usage: $0 --model-key KEY [options]

Model keys: llama, deepseek, qwen, qwen3_5, gemma3, gpt_oss, mistral, phi3, olmo3, olmoe

Options:
  --model-id ID           Override Hugging Face model id or local model path
  --input-file PATH       Override prompt file
  --output-length N       Override generated token count
  --batch-size N          Override batch size
  --summary-file PATH     Override summary file
  --log-dir PATH          Profiler and nsys log directory
  --skip-baseline         Do not run PyTorch baseline
  --skip-cutile           Do not run TileGym cuTile benchmark
  --skip-coverage         Do not run nsys kernel coverage
  -h, --help              Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-key) MODEL_KEY="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --input-file) INPUT_FILE="$2"; shift 2 ;;
        --output-length) OUTPUT_LENGTH="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --summary-file) SUMMARY_FILE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --skip-baseline) RUN_BASELINE=0; shift ;;
        --skip-cutile) RUN_CUTILE=0; shift ;;
        --skip-coverage) RUN_COVERAGE=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [ -z "${MODEL_KEY}" ]; then
    echo "Error: --model-key is required."
    usage
    exit 1
fi

case "${MODEL_KEY}" in
    llama)
        DEFAULT_MODEL_ID="meta-llama/Meta-Llama-3.1-8B"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_32K.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="llama_benchmark_summary.txt"
        TITLE="LLaMA-3.1-8B"
        ;;
    deepseek)
        DEFAULT_MODEL_ID="deepseek-ai/DeepSeek-V2-Lite-Chat"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=100
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="deepseek_benchmark_summary.txt"
        TITLE="DeepSeek-V2-Lite"
        ;;
    qwen)
        DEFAULT_MODEL_ID="Qwen/Qwen2-7B"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=16
        DEFAULT_SUMMARY_FILE="qwen_benchmark_summary.txt"
        TITLE="Qwen2-7B"
        ;;
    qwen3_5)
        DEFAULT_MODEL_ID="Qwen/Qwen3.5-0.8B"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="qwen3_5_benchmark_summary.txt"
        TITLE="Qwen3.5-0.8B"
        if [ "${LOG_DIR}" = "/logs" ]; then
            LOG_DIR="${TMPDIR:-/tmp}/tilegym_bench"
        fi
        ;;
    gemma3)
        DEFAULT_MODEL_ID="google/gemma-3-4b-it"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=100
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="gemma3_benchmark_summary.txt"
        TITLE="Gemma3"
        ;;
    gpt_oss)
        DEFAULT_MODEL_ID="openai/gpt-oss-20b"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=100
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="gpt_oss_benchmark_summary.txt"
        TITLE="GPT-OSS"
        ;;
    mistral)
        DEFAULT_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_32K.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="mistral_benchmark_summary.txt"
        TITLE="Mistral-7B"
        ;;
    phi3)
        DEFAULT_MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="phi3_benchmark_summary.txt"
        TITLE="Phi-3-mini-4k-instruct"
        ;;
    olmo3)
        DEFAULT_MODEL_ID="allenai/Olmo-3-1025-7B"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_32K.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="${LOG_DIR}/olmo3_benchmark_summary.txt"
        TITLE="OLMo-3-1025-7B"
        ;;
    olmoe)
        DEFAULT_MODEL_ID="allenai/OLMoE-1B-7B-0924"
        DEFAULT_INPUT_FILE="${PROJECT_DIR}/sample_inputs/input_prompt_small.txt"
        DEFAULT_OUTPUT_LENGTH=50
        DEFAULT_BATCH_SIZE=1
        DEFAULT_SUMMARY_FILE="${LOG_DIR}/olmoe_benchmark_summary.txt"
        TITLE="OLMoE-1B-7B-0924"
        ;;
    *)
        echo "Unknown --model-key: ${MODEL_KEY}"
        usage
        exit 1
        ;;
esac

MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL_ID}}"
INPUT_FILE="${INPUT_FILE:-${DEFAULT_INPUT_FILE}}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-${DEFAULT_OUTPUT_LENGTH}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
SUMMARY_FILE="${SUMMARY_FILE:-${DEFAULT_SUMMARY_FILE}}"

mkdir -p "${LOG_DIR}"
if [ "$(dirname "${SUMMARY_FILE}")" != "." ]; then
    mkdir -p "$(dirname "${SUMMARY_FILE}")"
fi
rm -f "${SUMMARY_FILE}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

run_cli() {
    "${PYTHON_BIN}" -m tilegym_hf_bench._cli "$@"
}

echo "========================================"
echo "  ${TITLE} Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_ID}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo "Batch size: ${BATCH_SIZE}"
echo "Log dir: ${LOG_DIR}"
echo ""

if [ "${RUN_BASELINE}" -eq 1 ]; then
    echo "Running PyTorch baseline..."
    run_cli \
        --model_id "${MODEL_ID}" \
        --profile \
        --log_dir "${LOG_DIR}" \
        --sentence_file "${INPUT_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --output_length "${OUTPUT_LENGTH}" \
        --summary_file "${SUMMARY_FILE}"
fi

if [ "${RUN_CUTILE}" -eq 1 ]; then
    echo ""
    echo "Running TileGym CUTILE backend..."
    run_cli \
        --model_id "${MODEL_ID}" \
        --use_tilegym \
        --use_cutile \
        --use_attn \
        --profile \
        --log_dir "${LOG_DIR}" \
        --sentence_file "${INPUT_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --output_length "${OUTPUT_LENGTH}" \
        --summary_file "${SUMMARY_FILE}"
fi

echo ""
echo "========================================"
echo "  Benchmark Results"
echo "========================================"
if [ -f "${SUMMARY_FILE}" ]; then
    cat "${SUMMARY_FILE}"
else
    echo "Summary file not found."
fi
echo "========================================"

if [ "${RUN_COVERAGE}" -eq 1 ]; then
    echo ""
    echo "========================================"
    echo "  TileGym Kernel Coverage"
    echo "========================================"
    run_cli \
        --model_id "${MODEL_ID}" \
        --use_tilegym \
        --use_cutile \
        --use_attn \
        --report_kernel_coverage \
        --log_dir "${LOG_DIR}" \
        --sentence_file "${INPUT_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --output_length "${OUTPUT_LENGTH}"
    echo "========================================"
fi
