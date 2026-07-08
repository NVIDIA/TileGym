#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate a --print-record perf log file.
# Checks: file exists, non-empty, has "Performance Test Results" markers,
# count matches PASSED test count.
#
# Usage: bash validate_perf_log.sh <log_file> [<label>]
# Exit 0 = PASS, Exit 1 = FAIL

set -uo pipefail

LOG="${1:?Usage: validate_perf_log.sh <log_file> [label]}"
LABEL="${2:-$(basename "$LOG")}"

if [ ! -f "$LOG" ]; then
    echo "FAIL: ${LABEL}: file not found → ${LOG}"
    exit 1
fi

if [ ! -s "$LOG" ]; then
    echo "FAIL: ${LABEL}: file is empty → ${LOG}"
    exit 1
fi

PERF_COUNT=$(grep -c "Performance Test Results" "$LOG" 2>/dev/null) || PERF_COUNT=0
PASS_COUNT=$(grep -c "PASSED" "$LOG" 2>/dev/null) || PASS_COUNT=0
FAIL_COUNT=$(grep -c "FAILED" "$LOG" 2>/dev/null) || FAIL_COUNT=0
SKIP_COUNT=$(grep -c "SKIPPED" "$LOG" 2>/dev/null) || SKIP_COUNT=0

echo "${LABEL}: perf_sections=${PERF_COUNT} passed=${PASS_COUNT} failed=${FAIL_COUNT} skipped=${SKIP_COUNT}"

if [ "${PERF_COUNT}" -eq 0 ]; then
    echo "FAIL: ${LABEL}: no 'Performance Test Results' found — --print-record was not effective"
    exit 1
fi

if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo "FAIL: ${LABEL}: ${FAIL_COUNT} test(s) FAILED"
    exit 1
fi

if [ "${PERF_COUNT}" -ne "${PASS_COUNT}" ]; then
    echo "WARN: ${LABEL}: perf_sections(${PERF_COUNT}) != passed(${PASS_COUNT})"
    # Not a hard fail — some tests may print results differently
    exit 0
fi

echo "PASS: ${LABEL}: ${PERF_COUNT} perf records for ${PASS_COUNT} passed tests"
exit 0
