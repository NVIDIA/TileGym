#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate all required output files for a converted kernel.
# Usage: bash validate_kernel.sh <kernel_name>
# Exit 0 = PASS. Exit 1 = FAIL.

set -uo pipefail

KERNEL_NAME="${1:?Usage: validate_kernel.sh <kernel_name>}"

# There is no separate cutile-rs checkout / CUTILE_RS_ROOT anymore. The
# aggregated cutile_kernels crate (one libcutile_kernels.so) lives inside the
# tilegym tree; CUTILE_RS_KERNELS_DIR can override its location.
TILEGYM="${TILEGYM_PATH:?TILEGYM_PATH must be set to your tilegym checkout}"
KERNELS_DIR="${CUTILE_RS_KERNELS_DIR:-${TILEGYM}/src/tilegym/ops/cutile_rs/cutile_kernels}"
BASE="${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set}/${KERNEL_NAME}"

fail=0
warn=0

ok() { echo "  OK: $1"; }
fail_item() { echo "  FAIL: $1"; fail=$((fail + 1)); }
warn_item() { echo "  WARN: $1"; warn=$((warn + 1)); }

check_file() {
    local path="$1"
    local desc="$2"
    if [ -f "$path" ] && [ -s "$path" ]; then
        ok "$desc"
    else
        fail_item "$desc missing or empty -> $path"
    fi
}

check_agent_log() {
    local agent="$1"
    local required="$2"
    local p1="${BASE}/reports/agent_logs/agent_${agent}.md"
    local p2="${BASE}/reports/agent_${agent}.md"

    if [ -f "$p1" ] && [ -s "$p1" ]; then
        ok "agent_${agent} log (${p1})"
        return 0
    fi
    if [ -f "$p2" ] && [ -s "$p2" ]; then
        ok "agent_${agent} log (${p2}; legacy location accepted)"
        return 0
    fi

    if [ "$required" = "yes" ]; then
        fail_item "agent_${agent} log missing (checked reports/agent_logs and reports/)"
    else
        ok "agent_${agent} log not required for this path"
    fi
}

validate_perf_if_present() {
    local file="$1"
    local label="$2"
    local required="$3"
    local validator="${TILEGYM}/.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/scripts/validate_perf_log.sh"

    if [ ! -f "${BASE}/${file}" ]; then
        if [ "$required" = "yes" ]; then
            fail_item "${file} missing"
        else
            warn_item "${file} missing (optional)"
        fi
        return
    fi

    if bash "$validator" "${BASE}/${file}" "$label" 2>/dev/null; then
        ok "${file} validated"
    else
        fail_item "${file} validation failed"
    fi
}

extract_ffi_fn() {
    local file="$1"
    python3 - "$file" <<'PY' 2>/dev/null
import re
import sys

path = sys.argv[1]
try:
    text = open(path, encoding="utf-8").read()
except OSError:
    raise SystemExit(0)

pat = re.compile(
    r"pub\s+(?:unsafe\s+)?extern\s+\"C\"\s+fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
m = pat.search(text)
if m:
    print(m.group(1))
PY
}

wrapper_call_info() {
    local wrapper="$1"
    local ffi_fn="$2"
    python3 - "$wrapper" "$ffi_fn" <<'PY' 2>/dev/null
import ast
import sys

path, ffi = sys.argv[1], sys.argv[2]
try:
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text, filename=path)
except Exception:
    print("0 0")
    raise SystemExit(0)

constants = {}

def record_const(target, value):
    if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
        constants[target.id] = value.value

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            record_const(target, node.value)
    elif isinstance(node, ast.AnnAssign):
        record_const(node.target, node.value)

def const_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None

def is_direct_attr(node):
    return isinstance(node, ast.Attribute) and node.attr == ffi

def is_getattr_symbol(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and const_string(node.args[1]) == ffi
    )

def is_bind_kernel_symbol(node):
    if not isinstance(node, ast.Call):
        return False
    name = None
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    if name not in {"bind_kernel_function_cffi", "bind_kernel_function", "bind_kernel", "load_kernel_function"}:
        return False
    return any(const_string(arg) == ffi for arg in node.args)

def is_symbol_binding_expr(node):
    # This identifies expressions that produce a callable bound to the exported
    # FFI symbol. Binding alone is not a live call.
    return is_direct_attr(node) or is_getattr_symbol(node) or is_bind_kernel_symbol(node)

aliases = set()

for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and is_symbol_binding_expr(node.value):
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    elif isinstance(node, ast.AnnAssign) and is_symbol_binding_expr(node.value):
        if isinstance(node.target, ast.Name):
            aliases.add(node.target.id)

factories = set()

for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
        continue
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            value = child.value
            if isinstance(value, ast.Name) and value.id in aliases:
                factories.add(node.name)
            elif value is not None and is_symbol_binding_expr(value):
                factories.add(node.name)

def is_factory_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in factories
    )

for node in ast.walk(tree):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        continue
    value = node.value
    if is_factory_call(value):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)

def is_live_symbol_invocation(func):
    # Counts executable calls only:
    #   lib.cutile_x(...)
    #   getattr(lib, "cutile_x")(...)
    #   bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)  (binds; ffi/lib then call lib.cutile_x)
    #   alias(...)
    #   _lib_fn()(...) where _lib_fn returns a symbol-bound callable.
    # Does not count constants, comments, binding-helper calls by themselves,
    # or aliases/factories that are never invoked.
    return (
        is_direct_attr(func)
        or is_getattr_symbol(func)
        or is_bind_kernel_symbol(func)
        or is_factory_call(func)
        or (isinstance(func, ast.Name) and func.id in aliases)
    )

live_lines = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if is_live_symbol_invocation(node.func):
        live_lines.append(getattr(node, "lineno", 0))

if live_lines:
    print(len(live_lines), min(live_lines))
else:
    print("0 0")
PY
}

echo "=== Validating: ${KERNEL_NAME} ==="

if [ ! -d "$BASE" ]; then
    fail_item "kernel output directory missing -> ${BASE}"
    echo "==="
    echo "Result: ${fail} fail, ${warn} warn"
    echo "FAIL"
    exit 1
fi

echo "--- cutile-rs: kernel files ---"
check_file "${BASE}/kernel.rs" "kernel.rs"
check_file "${BASE}/ffi.rs" "ffi.rs"
check_file "${BASE}/reference/analysis.json" "analysis.json"

HAS_VARIANTS=$(BASE_DIR="$BASE" python3 - <<'PY' 2>/dev/null
import json
import os
base = os.environ["BASE_DIR"]
path = os.path.join(base, "reference", "analysis.json")
try:
    d = json.load(open(path))
except Exception:
    print("unknown")
    raise SystemExit
v = d.get("kernel_variants", [])
print("multi" if isinstance(v, list) and len(v) > 1 else "single")
PY
)

if [ "$HAS_VARIANTS" = "multi" ]; then
    BASE_DIR="$BASE" python3 - <<'PY' 2>/dev/null
import json
import os
import sys

base = os.environ["BASE_DIR"]
d = json.load(open(os.path.join(base, "reference", "analysis.json")))
missing = []
for v in d.get("kernel_variants", []):
    if not isinstance(v, dict):
        continue
    ir = v.get("reference_ir")
    if not ir:
        continue
    path = os.path.normpath(os.path.join(base, ir))
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print(f"  OK: per-variant IR: {ir}")
    else:
        print(f"  FAIL: per-variant IR missing: {ir} (resolved: {path})")
        missing.append(ir)
sys.exit(1 if missing else 0)
PY
    [ $? -eq 0 ] || fail=$((fail + 1))

    gen_count=$(ls "${BASE}/generated/generated_"*.mlir 2>/dev/null | wc -l)
    if [ "$gen_count" -gt 0 ]; then
        ok "${gen_count} per-variant generated IR files"
    else
        fail_item "no per-variant generated IR files in generated/"
    fi
else
    check_file "${BASE}/reference/reference.mlir" "reference IR (single variant)"
    check_file "${BASE}/generated/generated.mlir" "generated IR"
fi

C_RAN=no
[ -s "${BASE}/generated/diff_report.md" ] && C_RAN=yes
[ -s "${BASE}/reports/agent_c.md" ] && C_RAN=yes
[ -s "${BASE}/reports/agent_logs/agent_c.md" ] && C_RAN=yes
if [ "$C_RAN" = "yes" ]; then
    check_file "${BASE}/generated/diff_report.md" "IR diff report"
    check_agent_log c yes
else
    ok "IR diff report not required (Agent C not spawned on happy path)"
    check_agent_log c no
fi

check_file "${BASE}/reports/correctness.md" "correctness report"
check_file "${BASE}/reports/performance.md" "performance report"

check_agent_log a yes
check_agent_log b yes
check_agent_log d yes
check_agent_log e yes

F_RAN=no
ls "${BASE}"/reports/perf_investigation_*.md >/dev/null 2>&1 && F_RAN=yes
[ -s "${BASE}/reports/agent_f.md" ] && F_RAN=yes
[ -s "${BASE}/reports/agent_logs/agent_f.md" ] && F_RAN=yes
if [ "$F_RAN" = "yes" ]; then
    check_agent_log f yes
else
    ok "agent_f log not required (Agent F not spawned)"
fi

echo "--- cutile-rs: perf raw logs ---"
validate_perf_if_present "baseline_perf_cutile.txt" "baseline_perf_cutile" yes
validate_perf_if_present "baseline_perf_triton_tileir.txt" "baseline_perf_triton_tileir" no
validate_perf_if_present "baseline_perf.txt" "baseline_perf" no
validate_perf_if_present "perf_cutile_rs.txt" "perf_cutile_rs" yes
validate_perf_if_present "perf_cutile_py.txt" "perf_cutile_py" no

echo "--- cutile-rs: dtype generic (Rule 16) ---"
if [ -f "${BASE}/kernel.rs" ]; then
    if grep -q "E: ElementType" "${BASE}/kernel.rs"; then
        ok "kernel.rs uses <E: ElementType>"
    else
        fail_item "kernel.rs hardcodes dtype - use <E: ElementType> (Rule 16)"
    fi

    hardcoded_ptr=$(grep -nE '\*mut (f16|f32|bf16)' "${BASE}/kernel.rs" 2>/dev/null \
        | grep -vE '\b(rstd|Rstd|RSTD|mean|Mean|MEAN|var|Var|VAR|inv_std|InvStd)([_a-zA-Z0-9]*)?: *\*mut f32' \
        | grep -vE '\*mut (f16|f32|bf16) *[,>]' \
        | wc -l) || hardcoded_ptr=0
    if [ "${hardcoded_ptr}" -gt 0 ]; then
        fail_item "kernel entry has hardcoded *mut f16/*mut f32 - use *mut E"
    else
        ok "no hardcoded dtype in entry signature (norm stats f32 buffers allow-listed)"
    fi

    if [ -f "${BASE}/ffi.rs" ]; then
        if grep -qE '"f16"|"f32"|"bf16"|dtype' "${BASE}/ffi.rs"; then
            ok "ffi.rs passes dtype in generics"
        else
            fail_item "ffi.rs does not pass dtype string - kernel will not specialize"
        fi
    fi
fi

echo "--- cutile-rs: FFI registration / aggregated cdylib ---"

FFI_FN=""
if [ -f "${BASE}/ffi.rs" ]; then
    FFI_FN=$(extract_ffi_fn "${BASE}/ffi.rs" | head -1)
fi
if [ -z "$FFI_FN" ]; then
    fail_item "could not extract extern C FFI function from ffi.rs"
else
    ok "ffi symbol declared: ${FFI_FN}"
fi

# ffi.rs ABI (Rule 35): tensors cross as *const TensorDesc, unpacked via
# borrow_tensor / DevicePointer — see references/coding-rules.md Rule 35.
if [ -f "${BASE}/ffi.rs" ]; then
    if grep -qE '\*const[[:space:]]+TensorDesc' "${BASE}/ffi.rs"; then
        ok "ffi.rs crosses tensors as *const TensorDesc"
    else
        fail_item "ffi.rs does not use '*const TensorDesc' (tensors carry dtype/shape/strides in the descriptor now)"
    fi
    if grep -qE 'borrow_tensor|DevicePointer::from_cu_deviceptr' "${BASE}/ffi.rs"; then
        ok "ffi.rs unpacks descriptors via borrow_tensor / DevicePointer::from_cu_deviceptr"
    else
        fail_item "ffi.rs does not unpack via borrow_tensor::<E> or DevicePointer::from_cu_deviceptr"
    fi
    if grep -qE 'from_raw_parts|mem::forget|transmute' "${BASE}/ffi.rs"; then
        fail_item "ffi.rs uses op-level from_raw_parts / mem::forget / transmute (banned — only ffi_util::borrow_tensor may call from_raw_parts)"
    else
        ok "no op-level from_raw_parts / mem::forget / transmute in ffi.rs"
    fi
    if grep -qE '#\[unsafe\(no_mangle\)\]' "${BASE}/ffi.rs"; then
        ok "ffi.rs uses #[unsafe(no_mangle)] (Rust 2024 export)"
    else
        fail_item "ffi.rs missing #[unsafe(no_mangle)] (legacy #[no_mangle] is banned)"
    fi
fi

# ONE aggregated cutile_kernels crate registers every op via a `mod` that
# include!s the op's kernel.rs + ffi.rs. There is no monolithic cutile-ffi
# include and no per-op .so.
LIB_RS="${KERNELS_DIR}/src/lib.rs"
if grep -qE "mod[[:space:]]+${KERNEL_NAME%_kernel}([[:space:]]|\{)|include!\(.*${KERNEL_NAME%_kernel}_kernel/(kernel|ffi)\.rs" "$LIB_RS" 2>/dev/null; then
    ok "cutile_kernels/src/lib.rs registers ${KERNEL_NAME%_kernel} (mod + include!)"
else
    fail_item "cutile_kernels/src/lib.rs does not register ${KERNEL_NAME%_kernel} (expected 'mod <op> { include!(...) }' → ${LIB_RS})"
fi

# Deps must be crates.io PINNED (=0.2.0), NOT path deps to a cutile-rs checkout.
CARGO_TOML="${KERNELS_DIR}/Cargo.toml"
if [ -f "$CARGO_TOML" ]; then
    if grep -qE 'cutile[[:space:]]*=[[:space:]]*"=0\.2\.0"' "$CARGO_TOML"; then
        ok "cutile_kernels/Cargo.toml pins cutile = \"=0.2.0\" (crates.io, Rule 33)"
    else
        fail_item "cutile_kernels/Cargo.toml does not pin cutile = \"=0.2.0\" (crates.io PINNED deps required)"
    fi
    if grep -qE 'path[[:space:]]*=' "$CARGO_TOML"; then
        fail_item "cutile_kernels/Cargo.toml uses a path dependency (banned — deps are crates.io pins, no cutile-rs checkout)"
    else
        ok "cutile_kernels/Cargo.toml uses no path deps"
    fi
else
    fail_item "cutile_kernels/Cargo.toml missing -> ${CARGO_TOML}"
fi

# The single aggregated cdylib is libcutile_kernels.so; there is no per-op .so
# and no libcutile_ffi.so.
SO_FOUND=$(find "${KERNELS_DIR}/target/release" "${TILEGYM}/src/tilegym/ops/cutile_rs" \
              -name libcutile_kernels.so 2>/dev/null | head -1)
if [ -n "$SO_FOUND" ]; then
    ok "aggregated cdylib found: ${SO_FOUND}"
else
    fail_item "libcutile_kernels.so not found (build with: cd ${KERNELS_DIR} && cargo build --release)"
fi

if [ -n "$FFI_FN" ] && [ -n "$SO_FOUND" ]; then
    nm_match=$(nm -D "${SO_FOUND}" 2>/dev/null | grep " T ${FFI_FN}" || true)
    if [ -n "$nm_match" ]; then
        ok "libcutile_kernels.so exports ${FFI_FN}"
    else
        fail_item "${FFI_FN} not exported by ${SO_FOUND}"
    fi
fi

# ffi_util.rs (shared TensorDesc / borrow_tensor / dtype_str) must be present.
if [ -f "${TILEGYM}/src/tilegym/ops/cutile_rs/ffi_util.rs" ]; then
    ok "shared ffi_util.rs present (TensorDesc / borrow_tensor / dtype_str)"
else
    warn_item "shared ffi_util.rs not found under ops/cutile_rs/ (expected for the aggregated crate)"
fi

PIPELINE=""
for p in "${BASE}/${KERNEL_NAME}_pipeline.rs" \
         "${BASE}/${KERNEL_NAME%_kernel}_pipeline.rs"; do
    [ -f "$p" ] && PIPELINE="$p" && break
done
if [ -n "$PIPELINE" ]; then
    ok "compile test $(basename "${PIPELINE}")"
else
    warn_item "compile test not found"
fi

echo "--- tilegym: wrapper ---"

WRAPPER=""
if [ -n "$FFI_FN" ]; then
    WRAPPER=$(grep -rl "${FFI_FN}" "${TILEGYM}/src/tilegym/ops/cutile_rs/"*.py 2>/dev/null | head -1)
fi
if [ -z "$WRAPPER" ]; then
    short="${KERNEL_NAME%_kernel}"
    for candidate in "${short}" "${KERNEL_NAME}" "${short//_/}"; do
        for f in "${TILEGYM}/src/tilegym/ops/cutile_rs/"*.py; do
            [ -f "$f" ] || continue
            basename=$(basename "$f" .py)
            if echo "${basename}" | grep -qi "${candidate}" 2>/dev/null; then
                WRAPPER="$f"
                break 2
            fi
        done
    done
fi

if [ -z "$WRAPPER" ] || [ ! -f "$WRAPPER" ]; then
    fail_item "tilegym wrapper not found for ${KERNEL_NAME}"
else
    ok "wrapper: $(basename "${WRAPPER}")"

    if grep -q 'register_impl.*cutile-rs' "${WRAPPER}"; then
        ok "@register_impl(cutile-rs)"
    else
        fail_item "wrapper missing @register_impl(..., backend='cutile-rs')"
    fi

    if [ -n "$FFI_FN" ]; then
        call_info="$(wrapper_call_info "${WRAPPER}" "${FFI_FN}")"
        call_sites=0
        first_call=0
        if [ -n "$call_info" ]; then
            read -r call_sites first_call <<< "$call_info"
        fi

        if [ "${call_sites:-0}" -gt 0 ]; then
            ok "wrapper executes ${FFI_FN} (${call_sites} live call site(s))"
        else
            fail_item "wrapper never executes ${FFI_FN} - dead code / fallback"
        fi

        if [ "${first_call:-0}" -gt 1 ]; then
            early=$(head -n "$((first_call - 1))" "${WRAPPER}" \
                | grep -cE '^\s*(return |raise )' 2>/dev/null) || early=0
            if [ "${early}" -gt 0 ]; then
                warn_item "${early} early return/raise before FFI call - review for unsupported-case gates, not fallback"
            fi
        fi
    fi

    wrapper_mod=$(basename "${WRAPPER}" .py)
    if grep -q "from \. import ${wrapper_mod}" "${TILEGYM}/src/tilegym/ops/cutile_rs/__init__.py" 2>/dev/null; then
        ok "__init__.py imports ${wrapper_mod}"
    else
        fail_item "__init__.py missing 'from . import ${wrapper_mod}'"
    fi
fi

echo "--- tilegym: test ---"

TEST_FILE=""
if [ -n "$WRAPPER" ] && [ -f "$WRAPPER" ]; then
    op_name=$(sed -n 's/.*register_impl("\([^"]*\)".*/\1/p' "${WRAPPER}" 2>/dev/null | head -1)
    if [ -n "$op_name" ]; then
        for candidate in "${op_name}" "${op_name//_act_/_activation_}" "${op_name}_kernel"; do
            f="${TILEGYM}/tests/ops/test_${candidate}.py"
            [ -f "$f" ] && TEST_FILE="$f" && break
        done
    fi
fi
if [ -z "$TEST_FILE" ]; then
    short="${KERNEL_NAME%_kernel}"
    for candidate in "${short}" "${short//_act_/_activation_}" "${short//_/}"; do
        f="${TILEGYM}/tests/ops/test_${candidate}.py"
        [ -f "$f" ] && TEST_FILE="$f" && break
    done
fi

if [ -z "$TEST_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    fail_item "test file not found"
else
    if grep -q 'cutile.rs\|cutile-rs' "${TEST_FILE}"; then
        ok "test: $(basename "${TEST_FILE}") has cutile-rs"
    else
        fail_item "$(basename "${TEST_FILE}") missing cutile-rs backend"
    fi
fi

echo "==="
echo "Result: ${fail} fail, ${warn} warn"
if [ ${fail} -eq 0 ]; then
    echo "PASS"
    exit 0
else
    echo "FAIL"
    exit 1
fi
