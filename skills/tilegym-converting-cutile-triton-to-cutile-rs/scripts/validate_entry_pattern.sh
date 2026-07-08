#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
#
# validate_entry_pattern.sh <kernel_name>
#
# Mechanically checks that kernel.rs #[cutile::entry] signatures match the
# load/store ops in structural reference.mlir files. The rule:
#
#   reference.mlir uses load_view_tko/store_view_tko_mut (TMA-style)
#     -> kernel.rs entry MUST use &Tensor<E, {[-1, ..., -1]}>
#
#   reference.mlir uses load_ptr_tko/store_ptr_tko (pointer-scatter)
#     -> kernel.rs entry MUST use *mut E
#
# Multi-variant kernels match reference_<variant>.mlir to the entry fn whose
# name contains <variant>. Dtype supplement IRs are not structural variants:
# analysis-declared supplements such as reference_ir_f32_supplement, or dtype
# suffix siblings such as reference_non_persistent_f32.mlir next to
# reference_non_persistent.mlir, are ignored for entry matching and reported as
# dtype-lowering evidence.
#
# Exit 0 = PASS. Exit 1 = FAIL_FIXABLE with actionable fix.

set -uo pipefail
KERNEL="${1:?Usage: validate_entry_pattern.sh <kernel_name>}"

: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT must be set}"

BASE="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL}"
KERNEL_RS="${BASE}/kernel.rs"
REF_DIR="${BASE}/reference"

if [ ! -f "$KERNEL_RS" ]; then
    echo "FAIL: kernel.rs not found at $KERNEL_RS"
    exit 1
fi
if [ ! -d "$REF_DIR" ]; then
    echo "FAIL: reference dir not found at $REF_DIR"
    exit 1
fi

python3 - "$KERNEL_RS" "$REF_DIR" "$KERNEL" <<'PYEOF'
import glob
import json
import os
import re
import sys

kernel_rs, ref_dir, kernel_name = sys.argv[1:4]
base_dir = os.path.dirname(ref_dir)

def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

def classify_ir(mlir_path):
    """Returns (kind, tma_count, ptr_count) where kind is TMA/PTR/MIXED/UNKNOWN."""
    text = read_text(mlir_path)
    tma_pat = r"\b(load_view_tko|store_view_tko_mut)\b"
    ptr_pat = r"\b(load_ptr_tko|store_ptr_tko)\b"
    tma = len(re.findall(tma_pat, text))
    ptr = len(re.findall(ptr_pat, text))
    if tma > 0 and ptr > 0:
        return ("MIXED", tma, ptr)
    if tma > 0:
        return ("TMA", tma, 0)
    if ptr > 0:
        return ("PTR", 0, ptr)
    return ("UNKNOWN", 0, 0)

def analysis_supplement_paths():
    analysis = os.path.join(ref_dir, "analysis.json")
    if not os.path.exists(analysis):
        analysis = os.path.join(base_dir, "reference", "analysis.json")
    try:
        data = json.load(open(analysis, encoding="utf-8"))
    except Exception:
        return set(), set()

    paths = set()
    basenames = set()

    def norm_path(value):
        if os.path.isabs(value):
            p = os.path.normpath(value)
        else:
            p = os.path.normpath(os.path.join(base_dir, value))
        return p

    def walk(obj, in_supplement=False):
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, in_supplement or ("supplement" in str(key).lower()))
        elif isinstance(obj, list):
            for value in obj:
                walk(value, in_supplement)
        elif in_supplement and isinstance(obj, str) and obj.endswith(".mlir"):
            p = norm_path(obj)
            paths.add(p)
            basenames.add(os.path.basename(p))

    walk(data)
    return paths, basenames

supp_paths, supp_basenames = analysis_supplement_paths()

all_variant_mlirs = sorted(glob.glob(os.path.join(ref_dir, "reference_*.mlir")))
single_mlir = os.path.join(ref_dir, "reference.mlir")
variant_basenames = {os.path.basename(p) for p in all_variant_mlirs}
dtype_suffix = re.compile(r"^reference_(.+)_(f16|f32|bf16|fp8|tf32|float16|float32)\.mlir$")

def is_dtype_suffix_sibling(path):
    name = os.path.basename(path)
    m = dtype_suffix.match(name)
    if not m:
        return False
    structural_sibling = f"reference_{m.group(1)}.mlir"
    return structural_sibling in variant_basenames

def is_supplement(path):
    norm = os.path.normpath(path)
    name = os.path.basename(path)
    return norm in supp_paths or name in supp_basenames or is_dtype_suffix_sibling(path)

ignored_supplements = [p for p in all_variant_mlirs if is_supplement(p)]
variant_mlirs = [p for p in all_variant_mlirs if not is_supplement(p)]

if variant_mlirs:
    ref_mlirs = variant_mlirs
elif os.path.exists(single_mlir):
    ref_mlirs = [single_mlir]
else:
    print(f"FAIL: no structural reference IR (.mlir) found in {ref_dir}")
    print("      Agent A must produce reference.mlir or structural reference_<variant>.mlir")
    if ignored_supplements:
        print("      Ignored supplement IRs cannot be the only structural references:")
        for p in ignored_supplements:
            print(f"        - {os.path.basename(p)}")
    sys.exit(1)

ir_classes = {}
for m in ref_mlirs:
    ir_classes[os.path.basename(m)] = classify_ir(m)

text = read_text(kernel_rs)

def find_entries(src):
    """Yield (fn_name, params_str). params_str excludes the outer parens."""
    i, n = 0, len(src)
    while i < n:
        m = re.search(r"#\[cutile::entry", src[i:])
        if not m:
            return
        i += m.end()
        depth = 1
        while i < n and depth > 0:
            c = src[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            i += 1
        fn_m = re.match(r"\s*(?:pub\s+)?unsafe\s+fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(", src[i:])
        if not fn_m:
            continue
        name = fn_m.group(1)
        i += fn_m.end()
        depth = 1
        start = i
        while i < n and depth > 0:
            c = src[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        params = src[start:i-1]
        yield (name, params)

entries = list(find_entries(text))
if not entries:
    print(f"FAIL: no #[cutile::entry] function found in {kernel_rs}")
    sys.exit(1)

def classify_entry(params):
    has_tensor = bool(re.search(r"&\s*Tensor\s*<", params))
    has_eptr = bool(re.search(r"\*\s*mut\s+E\b", params))
    if has_tensor and has_eptr:
        kind = "mixed"
    elif has_tensor:
        kind = "tensor"
    elif has_eptr:
        kind = "ptr"
    else:
        kind = "unknown"
    return (kind, has_tensor, has_eptr)

entry_classes = {n: classify_entry(p) for n, p in entries}

def expected_for(ir_kind):
    return {"TMA": "tensor", "PTR": "ptr"}.get(ir_kind)

mismatches = []

if len(ref_mlirs) == 1:
    ir_b = os.path.basename(ref_mlirs[0])
    ir_kind, tma_n, ptr_n = ir_classes[ir_b]
    expected = expected_for(ir_kind)
    if expected is None:
        print(f"WARN: reference IR {ir_b} has no recognized TMA / pointer-scatter ops")
        print(f"      (tma={tma_n}, ptr={ptr_n}) - skipping pattern check")
        sys.exit(0)
    for name, (kind, _, _) in entry_classes.items():
        if kind != expected:
            mismatches.append((name, kind, expected, ir_b, ir_kind, tma_n, ptr_n))
else:
    matched_entries = set()
    for mlir in ref_mlirs:
        ir_b = os.path.basename(mlir)
        variant_m = re.match(r"reference_(.+)\.mlir$", ir_b)
        if not variant_m:
            continue
        variant = variant_m.group(1).lower()
        v_tokens = variant.split("_")
        v_abbrev = "".join(t[0] for t in v_tokens) if len(v_tokens) > 1 else None
        candidates = []
        for ename in entry_classes:
            el = ename.lower()
            if variant in el:
                candidates.append((ename, "full"))
            elif v_abbrev and (f"_{v_abbrev}_" in f"_{el}_" or el.endswith(f"_{v_abbrev}") or f"_{v_abbrev}" in el):
                candidates.append((ename, "abbrev"))
            elif all(t in el for t in v_tokens):
                candidates.append((ename, "all_tokens"))
        ir_kind, tma_n, ptr_n = ir_classes[ir_b]
        expected = expected_for(ir_kind)
        if expected is None:
            continue
        if not candidates:
            mismatches.append((f"<no entry matches variant {variant}>", "NOT_FOUND", expected, ir_b, ir_kind, tma_n, ptr_n))
            continue
        for ename, _how in candidates:
            matched_entries.add(ename)
            kind, _, _ = entry_classes[ename]
            if kind != expected:
                mismatches.append((ename, kind, expected, ir_b, ir_kind, tma_n, ptr_n))

def fix_message(expected):
    if expected == "tensor":
        return (
            "    - For TMA-style entries, change params from `*mut E` to "
            "`&Tensor<E, {[-1, ..., -1]}>`\n"
            "    - Remove `make_tensor_view` calls from entry body - the macro builds\n"
            "      the view automatically from the &Tensor param.\n"
            "    - FFI builds `Arc<Tensor<E>>` from raw parts (Rule 12 branch a)\n"
            "    - Outputs use `partition_full_mut` on `&Tensor`, NEVER `&mut Tensor`"
        )
    if expected == "ptr":
        return (
            "    - For pointer-scatter entries, change params from `&Tensor` to `*mut E`\n"
            "    - Use iota + mask + offset_tile + load_ptr_tko in entry body\n"
            "    - No `make_tensor_view` (Rule 12 branch b)"
        )
    return "    - See coding-rules.md Rule 12 for the decision tree."

if not mismatches:
    print("OK: entry pattern matches structural reference IR")
    for name, (kind, _, _) in entry_classes.items():
        print(f"  entry `{name}` -> {kind}")
    for b, (k, t, p) in ir_classes.items():
        print(f"  IR {b} -> {k} (tma_ops={t}, ptr_ops={p})")
    for p in ignored_supplements:
        print(f"  supplement IR {os.path.basename(p)} ignored for entry matching; use for dtype lowering checks")
    sys.exit(0)

print("FAIL_FIXABLE: entry pattern mismatch with structural reference IR")
print()
seen_expected = set()
for name, actual, expected, ir_b, ir_kind, tma_n, ptr_n in mismatches:
    print(f"  Entry `{name}`:")
    print(f"    actual:   {actual}")
    print(f"    expected: {expected}  (reference {ir_b} uses {ir_kind} ops: tma={tma_n}, ptr={ptr_n})")
    print()
    seen_expected.add(expected)

if ignored_supplements:
    print("  Dtype supplement IRs ignored for entry matching:")
    for p in ignored_supplements:
        print(f"    - {os.path.basename(p)}")
    print("  Do not create fake dtype entries for these files; use them for dtype lowering checks.")
    print()

print("  Fix:")
for exp in sorted(seen_expected):
    print(fix_message(exp))
print()
print("  See: references/coding-rules.md Rule 12 (TMA vs pointer-scatter decision tree)")
sys.exit(1)
PYEOF
EXIT_CODE=$?
exit $EXIT_CODE
