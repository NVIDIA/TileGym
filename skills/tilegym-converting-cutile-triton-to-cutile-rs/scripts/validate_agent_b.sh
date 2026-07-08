#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate Agent B outputs: kernel.rs, ffi.rs, generated IR, dtype generic, compile test.

set -uo pipefail
KERNEL="${1:?Usage: validate_agent_b.sh <kernel_name>}"

# Per-kernel working dir (eval harness depends on it). There is no separate
# cutile-rs checkout / CUTILE_RS_ROOT anymore — the aggregated cutile_kernels
# crate lives under $TILEGYM_PATH/src/tilegym/ops/cutile_rs/.
: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set (e.g. $ISOLATED_CWD/cutile_kernel_out)}"

BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"
fail=0

check() { if [ -f "$1" ] && [ -s "$1" ]; then echo "OK: $2"; else echo "FAIL: $2 → $1"; fail=$((fail+1)); fi; }

check "${BASE}/kernel.rs" "kernel.rs"
# ffi.rs is NOT a B deliverable (device/host split): Agent D writes
# ffi.rs + wires the op into the aggregated cutile_kernels crate (builds the
# single libcutile_kernels.so), because the host launch path (output partition
# ABI, launch-grid validation, borrow_tensor / DevicePointer ownership) is only
# testable by D's pytest. B writes the device kernel + proves it with the
# in-Rust pipeline test.
# Generated IR: single-variant → generated.mlir, multi-variant → generated_{variant}.mlir
HAS_MULTI=$(python3 -c "
import json, os
f = '${BASE}/reference/analysis.json'
if not os.path.exists(f): print('single'); exit()
d = json.load(open(f))
v = d.get('kernel_variants', [])
print('multi' if isinstance(v, list) and len(v) > 1 else 'single')
" 2>/dev/null)
if [ "$HAS_MULTI" = "multi" ]; then
    gen_count=$(ls "${BASE}/generated/generated_"*.mlir 2>/dev/null | wc -l)
    if [ "$gen_count" -gt 0 ]; then
        echo "OK: ${gen_count} per-variant generated IR files"
    else
        echo "FAIL: multi-variant but no generated_{variant}.mlir files"
        fail=$((fail+1))
    fi
else
    check "${BASE}/generated/generated.mlir" "generated.mlir"
fi

# dtype generic
if [ -f "${BASE}/kernel.rs" ]; then
    if grep -q "E: ElementType" "${BASE}/kernel.rs"; then
        echo "OK: <E: ElementType>"
    else
        echo "FAIL: kernel.rs missing <E: ElementType> (Rule 16)"
        fail=$((fail+1))
    fi
fi

# (Rule 12): entry pattern must match reference IR
#   TMA-style IR (load_view_tko / store_view_tko_mut) → entry uses &Tensor
#   pointer-scatter IR (load_ptr_tko / store_ptr_tko) → entry uses *mut E
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "${SCRIPT_DIR}/validate_entry_pattern.sh" ] && [ -f "${BASE}/kernel.rs" ]; then
    if bash "${SCRIPT_DIR}/validate_entry_pattern.sh" "${KERNEL}" >&2; then
        echo "OK: entry pattern matches reference IR (Rule 12)"
    else
        echo "FAIL: entry pattern mismatch — see above (Rule 12)"
        fail=$((fail+1))
    fi
fi

# (multi-output transpose trap): a `&mut Tensor` OUTPUT param MUST carry
# CONCRETE tile entry shapes (e.g. {[1, TILE_H, TILE_D]}, {[TM, TN]}), NEVER -1
# wildcards. cutile-rs's launcher infers the output-partition grid from the
# param's tile shape; a -1 on a `&mut Tensor` output makes the host
# `(&mut t).partition([...])` fail with "partition shape mismatch. Expected
# [-1,...], got [...]" → EVERY config FAILs (the multi-output transpose trap, D fail_class:
# kernel). `-1` wildcards are valid ONLY on read-only `&Tensor` inputs.
if [ -f "${BASE}/kernel.rs" ]; then
    if grep -nE '&mut[[:space:]]+Tensor<[^;{]*\{[[:space:]]*\[[^]]*-1' "${BASE}/kernel.rs" >&2; then
        echo "FAIL: mutable output '&mut Tensor<.. {[..-1..]}>' uses a -1 wildcard shape (Rule 12 — transpose trap)."
        echo "      Mutable outputs MUST declare concrete tile entry shapes (e.g. {[1, TILE_H, TILE_D]});"
        echo "      cutile-rs rejects -1-shaped mutable outputs (output-partition grid cannot be inferred)."
        fail=$((fail+1))
    else
        echo "OK: no -1-wildcard mutable-output Tensor (multi-output transpose trap guard)"
    fi
fi

# agent log
check "${BASE}/reports/agent_logs/agent_b.md" "agent_b.md log"

# build_log.md mandatory + must contain CODE SNIPPETS for each failed attempt
# (not just error code/message — the actual source lines that triggered the error).
BUILD_LOG="${BASE}/reports/build_log.md"
check "${BUILD_LOG}" "build_log.md"
if [ -f "$BUILD_LOG" ] && [ -s "$BUILD_LOG" ]; then
    # Must mention at least one attempt
    if ! grep -qiE "^##? +Attempt|^Attempt [0-9]+" "$BUILD_LOG"; then
        echo "FAIL: build_log.md missing '## Attempt N' section header"
        fail=$((fail+1))
    fi
    # If any attempt FAILED, that attempt MUST include a fenced rust code block
    # showing the source snippet that triggered the error (not just error text).
    fail_count=$(grep -ciE "Result:[[:space:]]*FAIL|FAIL[[:space:]]*\(" "$BUILD_LOG" || true)
    if [ "$fail_count" -gt 0 ]; then
        FENCE_RUST='```rust'
        snippet_count=$(grep -cF "$FENCE_RUST" "$BUILD_LOG" || true)
        if [ "$snippet_count" -lt 1 ]; then
            echo "FAIL: build_log.md has $fail_count failed attempt(s) but no fenced rust code block (\`\`\`rust)"
            echo "      Agent B must paste the actual kernel.rs / ffi.rs lines that triggered each rustc error"
            fail=$((fail+1))
        fi
    fi
fi

# Host-side checks (ffi.rs, cutile_kernels build, wrapper, _FFI_CDEF, backend
# registration) are Agent D's, validated by validate_agent_d.sh — see the
# device/host split note near the kernel.rs check above.

# ─────────────────────────────────────────────────────────────────
# Agent B LANE GUARD (device/host split): B writes the device kernel
# only. ffi.rs, the Python wrapper, backend registration, and test parametrization
# are Agent D's. validate_agent_b.sh runs at the END of B's turn, BEFORE Agent D,
# so NONE of D's artifacts should exist yet. If they do, B overreached → hard FAIL.
# ─────────────────────────────────────────────────────────────────
# 0. ffi.rs is now Agent D's (host launch boundary). B must not write it.
if [ -f "${BASE}/ffi.rs" ]; then
    echo "FAIL: Agent B wrote ffi.rs (${BASE}/ffi.rs)."
    echo "      ffi.rs (C-ABI host launcher) is Agent D's deliverable now — D writes it,"
    echo "      wires it into the cutile_kernels crate (libcutile_kernels.so), and tests the"
    echo "      launch path via pytest. B writes kernel.rs only."
    echo "      Remove ffi.rs; keep the pipeline test (kernel-only) for the IR dump."
    fail=$((fail+1))
fi

TILE="${TILEGYM_PATH:-}"
if [ -n "$TILE" ] && [ -d "$TILE/src/tilegym" ]; then
    # 1. The cutile-rs wrapper module for THIS kernel must not exist yet (D creates it).
    for cand in "$TILE/src/tilegym/ops/cutile_rs/${KERNEL}.py" \
                "$TILE/src/tilegym/ops/cutile_rs/${KERNEL//_/}.py"; do
        if [ -f "$cand" ]; then
            echo "FAIL: Agent B wrote the Python wrapper ($cand)."
            echo "      Agent B is Rust-only (kernel.rs/ffi.rs). The wrapper is Agent D's deliverable."
            echo "      Remove it — Agent D creates the wrapper in its STEP 1.5."
            fail=$((fail+1))
        fi
    done

    # 2. The tilegym test file must not carry the cutile-rs parametrization yet (D adds it).
    TEST_FILE="$TILE/tests/ops/test_${KERNEL}.py"
    if [ -f "$TEST_FILE" ] && grep -qE '("cutile-rs"|cutile_rs)' "$TEST_FILE"; then
        echo "FAIL: Agent B wired cutile-rs into ${TEST_FILE}."
        echo "      Test-file parametrization is Agent D's job. Revert the edit."
        fail=$((fail+1))
    fi

    # 3. ops/cutile_rs/__init__.py must not register THIS kernel yet (D adds the import).
    RS_INIT="$TILE/src/tilegym/ops/cutile_rs/__init__.py"
    if [ -f "$RS_INIT" ] && grep -qE "(from[[:space:]]+\.[[:space:]]+import[[:space:]]+${KERNEL}([[:space:]]|,|$)|import[[:space:]]+${KERNEL}([[:space:]]|,|$))" "$RS_INIT"; then
        echo "FAIL: Agent B registered ${KERNEL} in ops/cutile_rs/__init__.py."
        echo "      Backend registration is Agent D's job. Revert the import."
        fail=$((fail+1))
    fi
else
    echo "WARN: TILEGYM_PATH unset — skipping Agent B lane guard"
fi

echo "---"
if [ $fail -eq 0 ]; then echo "VERDICT: COMPILED"; exit 0; else echo "VERDICT: FAIL_COMPILE"; exit 1; fi
