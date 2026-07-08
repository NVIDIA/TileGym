#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate Agent F outputs: perf_investigation_*.md with VERDICT first line.

set -uo pipefail
KERNEL="${1:?Usage: validate_agent_f.sh <kernel_name>}"

# Auto-resolve CUTILE_RS_ROOT from this script's location
: "${CUTILE_RS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set (e.g. $ISOLATED_CWD/cutile_kernel_out)}"

BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"
fail=0
count=0

for f in "${BASE}"/reports/perf_investigation_*.md; do
    [ -f "$f" ] || continue
    count=$((count+1))
    FIRST_LINE=$(head -1 "$f")
    if echo "$FIRST_LINE" | grep -qE "^VERDICT: (FIXABLE|ALIGNED|BLOCKED)$"; then
        echo "OK: $(basename $f): ${FIRST_LINE}"
    else
        echo "FAIL: $(basename $f): first line must be VERDICT: FIXABLE|ALIGNED|BLOCKED, got: ${FIRST_LINE}"
        fail=$((fail+1))
    fi
done

if [ $count -eq 0 ]; then
    echo "FAIL: no perf_investigation_*.md files found"
    fail=$((fail+1))
fi

# Agent log
if [ -f "${BASE}/reports/agent_logs/agent_f.md" ] && [ -s "${BASE}/reports/agent_logs/agent_f.md" ]; then
    echo "OK: agent_f.md log"
else
    echo "FAIL: agent_f.md log missing"
    fail=$((fail+1))
fi

echo "---"
if [ $fail -eq 0 ]; then echo "PASS"; exit 0; else echo "FAIL: ${fail} issue(s)"; exit 1; fi
