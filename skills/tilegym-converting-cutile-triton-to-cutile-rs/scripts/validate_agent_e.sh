#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate Agent E outputs: performance.md with VERDICT, raw perf logs.

set -uo pipefail
KERNEL="${1:?Usage: validate_agent_e.sh <kernel_name>}"

# Auto-resolve CUTILE_RS_ROOT from this script's location
: "${CUTILE_RS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set (e.g. $ISOLATED_CWD/cutile_kernel_out)}"

BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"
SCRIPTS_DIR="$(dirname "$0")"
REPORT="${BASE}/reports/performance.md"
fail=0

# performance.md exists with VERDICT
if [ ! -f "$REPORT" ] || [ ! -s "$REPORT" ]; then
    echo "FAIL: performance.md missing or empty"
    fail=$((fail+1))
else
    FIRST_LINE=$(head -1 "$REPORT")
    if echo "$FIRST_LINE" | grep -qE "^VERDICT: (DONE|INVESTIGATE)$"; then
        echo "OK: verdict line: ${FIRST_LINE}"
    else
        echo "FAIL: first line must be 'VERDICT: DONE' or 'VERDICT: INVESTIGATE', got: ${FIRST_LINE}"
        fail=$((fail+1))
    fi

    # Machine-readable geomean line — scorer's authoritative perf source.
    # Must be exactly: geomean_ratio=<number> (no spaces around '=', strictly
    # numeric: digits + optional single decimal fraction, no trailing 'x').
    # Exactly one such line is required.
    GM_COUNT=$(grep -cE "^geomean_ratio=[0-9]+(\.[0-9]+)?[[:space:]]*$" "$REPORT")
    if [ "$GM_COUNT" -eq 1 ]; then
        GM_LINE=$(grep -E "^geomean_ratio=" "$REPORT" | head -1)
        echo "OK: machine-readable geomean: ${GM_LINE}"
    elif [ "$GM_COUNT" -eq 0 ]; then
        echo "FAIL: performance.md missing required 'geomean_ratio=X.XXXX' line (scorer perf source)"
        fail=$((fail+1))
    else
        echo "FAIL: performance.md has ${GM_COUNT} 'geomean_ratio=' lines, expected exactly 1"
        fail=$((fail+1))
    fi
fi

# Raw perf logs via validate_perf_log.sh
for log in perf_cutile_rs.txt baseline_perf_cutile.txt; do
    if bash "${SCRIPTS_DIR}/validate_perf_log.sh" "${BASE}/${log}" "${log}" 2>/dev/null; then
        echo "OK: ${log}"
    else
        echo "FAIL: ${log}"
        fail=$((fail+1))
    fi
done

# Agent log
if [ -f "${BASE}/reports/agent_logs/agent_e.md" ] && [ -s "${BASE}/reports/agent_logs/agent_e.md" ]; then
    echo "OK: agent_e.md log"
else
    echo "FAIL: agent_e.md log missing"
    fail=$((fail+1))
fi

echo "---"
if [ $fail -eq 0 ]; then echo "PASS"; exit 0; else echo "FAIL: ${fail} issue(s)"; exit 1; fi
