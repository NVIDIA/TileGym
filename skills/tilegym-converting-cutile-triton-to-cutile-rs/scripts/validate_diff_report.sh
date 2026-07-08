#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate that diff_report.md meets all structural requirements.
# Usage: bash validate_diff_report.sh <kernel_name>
# Exit 0 = valid. Exit 1 = missing sections or content.
#
# Agent C MUST run this before reporting done.

KERNEL_NAME="${1:?Usage: validate_diff_report.sh <kernel_name>}"

# CUTILE_RS_ROOT must be set (validated by preflight.sh). The script lives in
# tilegym/.agents/skills/..., so any auto-resolve from its location would point
# at the tilegym repo, not the cutile-rs checkout — that's why we require it.
: "${CUTILE_RS_ROOT:?CUTILE_RS_ROOT not set. Run scripts/preflight.sh first or export CUTILE_RS_ROOT=/abs/path/to/cutile-rs.}"

REPORT="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL_NAME}/generated/diff_report.md"
LOG="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL_NAME}/reports/agent_logs/agent_c.md"

errors=0
warns=0

fail() { echo "FAIL: $1"; errors=$((errors + 1)); }
warn() { echo "WARN: $1"; warns=$((warns + 1)); }
ok()   { echo "  OK: $1"; }

echo "Validating diff_report.md for: ${KERNEL_NAME}"
echo "==="

# ---- File existence ----
if [ ! -f "${REPORT}" ]; then
    fail "diff_report.md not found at ${REPORT}"
    echo "==="
    echo "FAIL: ${errors} error(s)"
    exit 1
fi

if [ ! -f "${LOG}" ]; then
    warn "agent_c.md log not found at ${LOG}"
fi

# ---- Step 1: Structural Comparison (op count table) ----
echo "--- Step 1: Structural Comparison ---"
if grep -qi "structural comparison\|op count\|step 1\|diff_ir" "${REPORT}"; then
    ok "Step 1 section found"
    # Must have a table with | Op | or | # |
    if grep -qE "^\|.*\|.*\|.*\|" "${REPORT}"; then
        ok "Contains table"
    else
        fail "Step 1 has no table (expected | Op | Reference | Generated | Match |)"
    fi
else
    fail "Step 1 (Structural Comparison / op count) section missing"
fi

# ---- Step 2: Checklist Diff ----
echo "--- Step 2: Checklist Diff ---"
if grep -qi "checklist\|step 2\|item-by-item" "${REPORT}"; then
    ok "Step 2 section found"
    # Must have Status column values
    if grep -qiE "MATCH|DIFFERS|MISSING|SEMANTIC_EQUIV" "${REPORT}"; then
        ok "Contains status values (MATCH/DIFFERS/MISSING/SEMANTIC_EQUIV)"
    else
        fail "Step 2 table has no status values"
    fi
    # Must have Severity column values
    if grep -qiE "CRITICAL|WARN|INFO" "${REPORT}"; then
        ok "Contains severity values"
    else
        warn "Step 2 table may be missing severity column"
    fi
else
    fail "Step 2 (Checklist Diff) section missing"
fi

# ---- Step 3: Root Cause Analysis ----
echo "--- Step 3: Root Cause Analysis ---"
if grep -qi "root cause\|step 3" "${REPORT}"; then
    ok "Step 3 section found"
    # Must classify each difference
    if grep -qiE "user_code|dsl_limitation|compiler_bug|semantic_equiv" "${REPORT}"; then
        ok "Contains root cause categories"
    else
        warn "Step 3 may be missing root cause classifications"
    fi
else
    fail "Step 3 (Root Cause Analysis) section missing"
fi

# ---- Step 4: Impact Assessment ----
echo "--- Step 4: Impact Assessment ---"
if grep -qi "impact\|step 4" "${REPORT}"; then
    ok "Step 4 section found"
    if grep -qiE "correctness|performance|blocking" "${REPORT}"; then
        ok "Contains impact types"
    else
        warn "Step 4 may be missing impact assessment"
    fi
else
    fail "Step 4 (Impact Assessment) section missing"
fi

# ---- Step 5: Verdict ----
echo "--- Step 5: Verdict ---"
if grep -qi "verdict\|step 5" "${REPORT}"; then
    ok "Step 5 section found"
    # Must have exactly one of the four verdicts
    verdict_count=0
    for v in "PASS_WITH_NOTES" "PASS" "FAIL_FIXABLE" "FAIL_COMPILER_BUG"; do
        if grep -q "${v}" "${REPORT}"; then
            verdict_count=$((verdict_count + 1))
            echo "  OK: Verdict found: ${v}"
        fi
    done
    if [ ${verdict_count} -eq 0 ]; then
        fail "No verdict found (expected PASS / PASS_WITH_NOTES / FAIL_FIXABLE / FAIL_COMPILER_BUG)"
    fi

    # If FAIL_FIXABLE, must have action items
    if grep -q "FAIL_FIXABLE" "${REPORT}"; then
        if grep -qiE "action item|fix.*agent b|agent b.*fix|what.*fix" "${REPORT}"; then
            ok "FAIL_FIXABLE has action items for Agent B"
        else
            fail "FAIL_FIXABLE verdict but no action items for Agent B"
        fi
    fi

    # If PASS_WITH_NOTES, must list known gaps
    if grep -q "PASS_WITH_NOTES" "${REPORT}"; then
        if grep -qiE "known gap|accepted|limitation" "${REPORT}"; then
            ok "PASS_WITH_NOTES lists known gaps"
        else
            warn "PASS_WITH_NOTES but no known gaps listed"
        fi
    fi
else
    fail "Step 5 (Verdict) section missing"
fi

# ---- Key checks from diff_ir.sh ----
echo "--- Key attribute checks ---"

# assume div_by coverage
if grep -qiE "assume.*div_by|assume_div" "${REPORT}"; then
    ok "assume div_by analysis present"
else
    warn "No mention of assume div_by alignment analysis"
fi

# flush_to_zero / rounding<approx> coverage
if grep -qiE "flush_to_zero|FTZ|rounding.*approx" "${REPORT}"; then
    ok "FTZ / rounding analysis present"
else
    warn "No mention of flush_to_zero or rounding<approx> analysis"
fi

# strides analysis
if grep -qiE "strides.*\[.*,.*1\]|strides=\[" "${REPORT}"; then
    ok "Strides analysis present"
else
    warn "No mention of strides=[?,1] analysis"
fi

# ---- Summary ----
echo "==="
if [ ${errors} -eq 0 ]; then
    if [ ${warns} -gt 0 ]; then
        echo "PASS (with ${warns} warning(s))"
    else
        echo "PASS"
    fi
    exit 0
else
    echo "FAIL: ${errors} error(s), ${warns} warning(s)"
    exit 1
fi
