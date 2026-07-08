#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate Agent D outputs: correctness.md with VERDICT first line.

set -uo pipefail
KERNEL="${1:?Usage: validate_agent_d.sh <kernel_name>}"

# Per-kernel working dir (eval harness depends on it). There is no separate
# cutile-rs checkout / CUTILE_RS_ROOT anymore — the aggregated cutile_kernels
# crate lives under $TILEGYM_PATH/src/tilegym/ops/cutile_rs/cutile_kernels and
# builds ONE libcutile_kernels.so.
: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set (e.g. $ISOLATED_CWD/cutile_kernel_out)}"

BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"
REPORT="${BASE}/reports/correctness.md"
fail=0

if [ ! -f "$REPORT" ] || [ ! -s "$REPORT" ]; then
    echo "FAIL: correctness.md missing or empty"
    exit 1
fi

# First line must be VERDICT:
FIRST_LINE=$(head -1 "$REPORT")
if echo "$FIRST_LINE" | grep -qE "^VERDICT: (ALL_PASS|FAIL)$"; then
    echo "OK: verdict line: ${FIRST_LINE}"
else
    echo "FAIL: first line must be 'VERDICT: ALL_PASS' or 'VERDICT: FAIL', got: ${FIRST_LINE}"
    fail=$((fail+1))
fi

# Must have results table
if grep -q "| Test " "$REPORT"; then
    echo "OK: has results table"
else
    echo "FAIL: missing results table"
    fail=$((fail+1))
fi

# Raw pytest log — format checks (not just existence)
RAW_LOG="${BASE}/correctness_cutile_rs.txt"
if [ ! -f "$RAW_LOG" ]; then
    echo "FAIL: correctness_cutile_rs.txt missing"
    fail=$((fail+1))
elif [ ! -s "$RAW_LOG" ]; then
    echo "FAIL: correctness_cutile_rs.txt empty"
    fail=$((fail+1))
else
    RAW_PASS=$(grep -c "PASSED" "$RAW_LOG" 2>/dev/null) || RAW_PASS=0
    RAW_FAIL=$(grep -c "FAILED" "$RAW_LOG" 2>/dev/null) || RAW_FAIL=0
    RAW_SKIP=$(grep -c "SKIPPED" "$RAW_LOG" 2>/dev/null) || RAW_SKIP=0
    RAW_TOTAL=$((RAW_PASS + RAW_FAIL + RAW_SKIP))
    echo "OK: correctness_cutile_rs.txt ($(wc -l < "$RAW_LOG") lines, passed=${RAW_PASS} failed=${RAW_FAIL} skipped=${RAW_SKIP})"
    if [ "${RAW_TOTAL}" -eq 0 ]; then
        echo "FAIL: correctness_cutile_rs.txt has no PASSED/FAILED/SKIPPED — not valid pytest output"
        fail=$((fail+1))
    fi

    # ── Fake-pytest detection (v42 anti-pattern) ──
    # Real pytest output MUST contain at least one of these signature lines
    # printed by the pytest framework itself. If none are present, Agent D
    # was almost certainly running a hand-rolled python loop that mimics
    # pytest format strings — banned by agent_d.md HARD RULE.
    # Require markers that fake runners DON'T usually mimic. `collected N items`
    # and `test session starts` were mimicked by v42's fake runner — exclude them.
    # `platform `, `rootdir:`, `plugins:` are emitted ONLY by real pytest framework.
    if grep -qE "^platform [a-z]+ -- Python|^rootdir:|^plugins:" "$RAW_LOG"; then
        echo "OK: correctness_cutile_rs.txt contains real-pytest framework markers"
    else
        echo "FAIL: correctness_cutile_rs.txt has PASSED/FAILED lines but NO real-pytest framework markers"
        echo "      (no 'platform linux -- Python ...', 'rootdir: ...', 'plugins: ...')"
        echo "      → looks like a hand-rolled fake-pytest runner (banned, see agent_d.md HARD RULE)"
        fail=$((fail+1))
    fi
    # Also flag the specific v42 anti-pattern phrase if it leaks in
    if grep -qE "FAILED - max_diff=.*> 1e-2$|if diff > 1e-2" "$RAW_LOG"; then
        echo "FAIL: correctness_cutile_rs.txt contains hand-rolled abs-threshold judgment"
        echo "      ('max_diff > 1e-2' / 'if diff > 1e-2'). MUST use real pytest + common.py:compare_tensors."
        fail=$((fail+1))
    fi
fi

# Agent log
if [ -f "${BASE}/reports/agent_logs/agent_d.md" ] && [ -s "${BASE}/reports/agent_logs/agent_d.md" ]; then
    echo "OK: agent_d.md log"
else
    echo "FAIL: agent_d.md log missing"
    fail=$((fail+1))
fi

# ─────────────────────────────────────────────────────────────────
# ffi.rs + libcutile_kernels.so checks (device/host split): Agent D now writes
# ffi.rs (the C-ABI host launcher), wires the op into the aggregated
# cutile_kernels crate, and builds the single shared object. The launch path —
# output-partition ABI, launch-grid validation, borrow_tensor / DevicePointer
# ownership — is only exercised by D's pytest, so D owns it end-to-end.
# ─────────────────────────────────────────────────────────────────
BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"

# A. ffi.rs must exist (D wrote it)
if [ -f "${BASE}/ffi.rs" ] && [ -s "${BASE}/ffi.rs" ]; then
    echo "OK: ffi.rs present (Agent D)"
else
    echo "FAIL: ffi.rs missing — Agent D must write the C-ABI host launcher (${BASE}/ffi.rs)"
    fail=$((fail+1))
fi

# A'. Tensors must cross the FFI as *const TensorDesc (not raw ptr + dim/stride
# args), and the op must NOT reach for op-level from_raw_parts / mem::forget /
# transmute — unpacking is via borrow_tensor::<E> or DevicePointer::from_cu_deviceptr.
if [ -f "${BASE}/ffi.rs" ]; then
    if grep -qE '\*const[[:space:]]+TensorDesc' "${BASE}/ffi.rs"; then
        echo "OK: ffi.rs takes tensors as *const TensorDesc"
    else
        echo "FAIL: ffi.rs does not take tensors as '*const TensorDesc'"
        echo "      Tensors cross the boundary as descriptors now (dtype/shape/strides inside),"
        echo "      not raw ptr + loose dim/stride/dtype/elem_size args."
        fail=$((fail+1))
    fi
    if grep -qE 'borrow_tensor|DevicePointer::from_cu_deviceptr' "${BASE}/ffi.rs"; then
        echo "OK: ffi.rs unpacks descriptors via borrow_tensor / DevicePointer::from_cu_deviceptr"
    else
        echo "FAIL: ffi.rs does not unpack the descriptor via borrow_tensor::<E> or DevicePointer::from_cu_deviceptr"
        fail=$((fail+1))
    fi
    if grep -qE 'from_raw_parts|mem::forget|transmute' "${BASE}/ffi.rs"; then
        echo "FAIL: ffi.rs uses op-level from_raw_parts / mem::forget / transmute"
        echo "      These are banned at op level: from_raw_parts lives only inside ffi_util::borrow_tensor,"
        echo "      and ManuallyDrop (borrow_tensor) is the ownership gate — no mem::forget/transmute needed."
        fail=$((fail+1))
    else
        echo "OK: no op-level from_raw_parts / mem::forget / transmute in ffi.rs"
    fi
    # Rust 2024 export + real device_id (not hardcoded Device::new(0)) + null-check.
    if grep -qE '#\[unsafe\(no_mangle\)\]' "${BASE}/ffi.rs"; then
        echo "OK: ffi.rs uses #[unsafe(no_mangle)] (Rust 2024 export)"
    else
        echo "FAIL: ffi.rs does not use #[unsafe(no_mangle)] (legacy #[no_mangle] is banned)"
        fail=$((fail+1))
    fi
    if grep -qE 'Device::new\([[:space:]]*device_id' "${BASE}/ffi.rs"; then
        echo "OK: ffi.rs builds Device from device_id (multi-GPU correct)"
    else
        echo "FAIL: ffi.rs does not build Device::new(device_id.max(0) as usize) — must not hardcode Device::new(0)"
        fail=$((fail+1))
    fi
fi

# B. the aggregated shared object (single libcutile_kernels.so) must exist and
# export this op's symbol cutile_{kernel}. The loader autobuilds the
# cutile_kernels crate; CUTILE_RS_KERNELS_DIR overrides its location.
KERNELS_DIR="${CUTILE_RS_KERNELS_DIR:-${TILEGYM_PATH:-}/src/tilegym/ops/cutile_rs/cutile_kernels}"
SO=$(find "${KERNELS_DIR}/target/release" "${TILEGYM_PATH:-}/src/tilegym/ops/cutile_rs" \
        -name libcutile_kernels.so 2>/dev/null | head -1)
if [ -n "$SO" ] && nm -D "$SO" 2>/dev/null | grep -q " T cutile_${KERNEL}$"; then
    echo "OK: libcutile_kernels.so ($SO) exports symbol cutile_${KERNEL}"
else
    echo "FAIL: libcutile_kernels.so missing or symbol cutile_${KERNEL} not exported"
    echo "      Agent D must register the op in cutile_kernels/src/lib.rs"
    echo "        mod ${KERNEL} { include!(\"../../${KERNEL}_kernel/kernel.rs\"); include!(\"../../${KERNEL}_kernel/ffi.rs\"); }"
    echo "      then build the single aggregated cdylib:"
    echo "        cd ${KERNELS_DIR} && cargo build --release"
    fail=$((fail+1))
fi

# C. FFI compile-option lever: if analysis.json tunes num_cta_in_cga, ffi.rs must
# expose+apply it (so the wrapper's autotuned value actually reaches codegen).
ANALYSIS="${BASE}/reference/analysis.json"
if [ -f "$ANALYSIS" ] && [ -f "${BASE}/ffi.rs" ]; then
    TUNES_CGA=$(python3 -c "
import json
d = json.load(open('${ANALYSIS}')); s = json.dumps(d)
print('yes' if ('num_cta_in_cga' in s or 'num_ctas' in s) else 'no')
" 2>/dev/null)
    if [ "$TUNES_CGA" = "yes" ]; then
        if grep -q "num_cta_in_cga" "${BASE}/ffi.rs"; then
            echo "OK: ffi.rs exposes/applies num_cta_in_cga compile-option"
        else
            echo "FAIL: analysis.json tunes num_cta_in_cga/num_ctas but ffi.rs does not apply it"
            echo "      Agent D must: if num_cta_in_cga > 0 { opts = opts.num_cta_in_cga(v); }"
            fail=$((fail+1))
        fi
    fi
fi

# ─────────────────────────────────────────────────────────────────
# Wrapper + backend-registration + cffi boundary + autotune checks.
# Agent D owns the Python wrapper and all tilegym backend wiring.
# ─────────────────────────────────────────────────────────────────
TILE="${TILEGYM_PATH:-}"

# Locate the wrapper D was supposed to write.
WRAPPER=""
if [ -n "$TILE" ]; then
    WRAPPER=$(find "$TILE/src/tilegym/ops/cutile_rs" -name "${KERNEL}.py" -o -name "${KERNEL//_/}.py" 2>/dev/null | head -1)
    if [ -z "$WRAPPER" ]; then
        WRAPPER=$(find "$TILE/src/tilegym/ops/cutile_rs" -name "*.py" -exec grep -l "register_impl.*${KERNEL}" {} \; 2>/dev/null | head -1)
    fi
fi

# 1. wrapper exists
if [ -n "$WRAPPER" ] && [ -f "$WRAPPER" ]; then
    echo "OK: cutile_rs wrapper present ($WRAPPER)"
else
    echo "FAIL: cutile_rs wrapper for ${KERNEL} not found under ops/cutile_rs/"
    echo "      Agent D must create src/tilegym/ops/cutile_rs/<name>.py (STEP 1.5)"
    fail=$((fail+1))
fi

# 2. cffi boundary — the wrapper must declare a _FFI_CDEF whose tensor args are
# `const TensorDesc*`, bind via bind_kernel_function_cffi, and pack tensors with
# make_tensor_desc. (This replaces the old ctypes _FFI_ARGTYPES list; the
# descriptor carries dtype/shape/strides, so there is no loose arg list to drift
# and no 32-bit pointer truncation risk.)
if [ -n "$WRAPPER" ] && [ -f "$WRAPPER" ]; then
    if grep -qE '_FFI_CDEF|bind_kernel_function_cffi' "$WRAPPER"; then
        echo "OK: wrapper uses cffi (_FFI_CDEF + bind_kernel_function_cffi)"
    else
        echo "FAIL: wrapper missing cffi boundary (_FFI_CDEF / bind_kernel_function_cffi)"
        echo "      cutile-rs wrappers are cffi now, not ctypes. Declare a _FFI_CDEF string whose"
        echo "      tensor args are 'const TensorDesc*', and bind with bind_kernel_function_cffi."
        fail=$((fail+1))
    fi
    if grep -qE '\bmake_tensor_desc\b' "$WRAPPER"; then
        echo "OK: wrapper packs tensors with make_tensor_desc"
    else
        echo "FAIL: wrapper does not pack tensors with make_tensor_desc (const TensorDesc* boundary)"
        fail=$((fail+1))
    fi
    if grep -qE '\bcheck_rc\b' "$WRAPPER"; then
        echo "OK: wrapper checks the FFI return code with check_rc"
    else
        echo "FAIL: wrapper does not call check_rc(rc, _FFI_NAME) after the launch"
        fail=$((fail+1))
    fi
    # Guard against a leftover ctypes argtypes list (old pattern).
    if grep -qE '_FFI_ARGTYPES|\.argtypes[[:space:]]*=' "$WRAPPER"; then
        echo "FAIL: wrapper still declares ctypes _FFI_ARGTYPES/.argtypes (old pattern)"
        echo "      Replace with the cffi _FFI_CDEF + make_tensor_desc boundary."
        fail=$((fail+1))
    fi
fi

# 3. autotune_launch when analysis.json carries autotune configs
ANALYSIS="${BASE}/reference/analysis.json"
if [ -f "$ANALYSIS" ] && [ -n "$WRAPPER" ] && [ -f "$WRAPPER" ]; then
    HAS_AUTOTUNE=$(python3 -c "
import json
d = json.load(open('${ANALYSIS}'))
at = d.get('autotune_configs', []); ab = d.get('autotune_backends', [])
print('yes' if (at and len(at) > 0) or (ab and len(ab) > 0) else 'no')
" 2>/dev/null)
    if [ "$HAS_AUTOTUNE" = "yes" ]; then
        if grep -q "autotune_launch" "$WRAPPER"; then
            echo "OK: wrapper uses autotune_launch (autotune_configs present)"
        else
            echo "FAIL: analysis.json has autotune_configs but wrapper does NOT use autotune_launch"
            echo "      Agent D must import from tilegym.backend.cutile_rs and call autotune_launch()"
            fail=$((fail+1))
        fi
    else
        echo "OK: no autotune required (analysis.json has no autotune_configs)"
    fi
fi

# 4. backend registration — 3 entry points or pytest sees 0 cutile-rs tests
if [ -n "$TILE" ] && [ -d "$TILE/src/tilegym" ]; then
    OPS_INIT="$TILE/src/tilegym/ops/__init__.py"
    if [ -f "$OPS_INIT" ] && grep -qE "(from\s+\.\s+import\s+cutile_rs|import.*cutile_rs)" "$OPS_INIT"; then
        echo "OK: ops/__init__.py imports cutile_rs"
    else
        echo "FAIL: ops/__init__.py does NOT import cutile_rs (Agent D must add 'from . import cutile_rs')"
        fail=$((fail+1))
    fi

    SELECTOR="$TILE/src/tilegym/backend/selector.py"
    if [ -f "$SELECTOR" ] && grep -qE '("cutile-rs"|cutile_rs)' "$SELECTOR"; then
        echo "OK: backend/selector.py references cutile-rs"
    else
        echo "FAIL: backend/selector.py does NOT register cutile-rs (Agent D must wire is_cutile_rs_available())"
        fail=$((fail+1))
    fi

    TEST_FILE="$TILE/tests/ops/test_${KERNEL}.py"
    if [ -f "$TEST_FILE" ] && grep -qE '("cutile-rs"|cutile_rs)' "$TEST_FILE"; then
        echo "OK: tests/ops/test_${KERNEL}.py includes cutile-rs backend"
    else
        echo "FAIL: tests/ops/test_${KERNEL}.py does NOT include cutile-rs (Agent D must add it to parametrization)"
        fail=$((fail+1))
    fi
else
    echo "WARN: TILEGYM_PATH unset or missing — skipping backend registration checks"
fi

echo "---"
if [ $fail -eq 0 ]; then echo "PASS"; exit 0; else echo "FAIL: ${fail} issue(s)"; exit 1; fi
