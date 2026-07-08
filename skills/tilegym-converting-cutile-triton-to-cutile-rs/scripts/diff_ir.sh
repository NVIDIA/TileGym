#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Compare two CUDA Tile IR files and emit a routing verdict.
#
# Usage: bash diff_ir.sh <reference.mlir> <generated.mlir>
# Exit 0 = PASS
# Exit 1 = FAIL_FIXABLE
# Exit 2 = PASS_WITH_NOTES

set -uo pipefail

REF="${1:?Usage: diff_ir.sh <reference.mlir> <generated.mlir>}"
GEN="${2:?Usage: diff_ir.sh <reference.mlir> <generated.mlir>}"

[ -f "$REF" ] || { echo "ERROR: $REF not found"; exit 1; }
[ -f "$GEN" ] || { echo "ERROR: $GEN not found"; exit 1; }

critical_fail=0
workaround_count=0
note_count=0
notes=""

grep_count() {
    local pattern="$1"
    local file="$2"
    local n
    n=$(grep -cE "$pattern" "$file" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$n" ]; then
        echo 0
    else
        echo "$n"
    fi
}

grep_count_fixed() {
    local pattern="$1"
    local file="$2"
    local n
    n=$(grep -cF "$pattern" "$file" 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$n" ]; then
        echo 0
    else
        echo "$n"
    fi
}

first_pcre() {
    local pattern="$1"
    local file="$2"
    local value
    value=$(grep -oP "$pattern" "$file" 2>/dev/null | head -1)
    if [ -z "$value" ]; then
        echo 0
    else
        echo "$value"
    fi
}

join_egrep() {
    local pattern="$1"
    local file="$2"
    grep -oE "$pattern" "$file" 2>/dev/null | sort -u | tr '\n' ' '
}

join_pcre() {
    local pattern="$1"
    local file="$2"
    grep -oP "$pattern" "$file" 2>/dev/null | sort -u | tr '\n' ' '
}

append_note() {
    local msg="$1"
    notes="${notes}\n${msg}"
    note_count=$((note_count + 1))
}

append_warn() {
    local msg="$1"
    append_note "WARN: ${msg}"
}

append_critical() {
    local msg="$1"
    critical_fail=$((critical_fail + 1))
    append_note "CRITICAL: ${msg}"
}

in_array() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [ "$needle" = "$item" ] && return 0
    done
    return 1
}

count_ops() {
    local file="$1"

    echo "load_ptr_tko $(grep_count_fixed 'load_ptr_tko' "$file")"
    echo "store_ptr_tko $(grep_count_fixed 'store_ptr_tko' "$file")"
    echo "load_view_tko $(grep_count_fixed 'load_view_tko' "$file")"
    echo "store_view_tko $(grep_count_fixed 'store_view_tko' "$file")"

    echo "reduce $(grep_count '^[[:space:]]*(%[A-Za-z0-9_]+ = )?reduce[[:space:]]' "$file")"
    echo "exp $(grep_count '(^|[[:space:]])exp([[:space:]]|$)' "$file")"
    echo "exp2 $(grep_count '(^|[[:space:]])exp2([[:space:]]|$)' "$file")"
    echo "rsqrt $(grep_count '(^|[[:space:]])rsqrt([[:space:]]|$)' "$file")"
    echo "fma $(grep_count '(^|[[:space:]])fma([[:space:]]|$)' "$file")"
    echo "mma $(grep_count '(^|[[:space:]])mmaf?([[:space:]]|$)' "$file")"

    echo "addf $(grep_count '(^|[[:space:]])addf([[:space:]]|$)' "$file")"
    echo "subf $(grep_count '(^|[[:space:]])subf([[:space:]]|$)' "$file")"
    echo "mulf $(grep_count '(^|[[:space:]])mulf([[:space:]]|$)' "$file")"
    echo "divf $(grep_count '(^|[[:space:]])divf([[:space:]]|$)' "$file")"
    echo "maxf $(grep_count '(^|[[:space:]])maxf([[:space:]]|$)' "$file")"
    echo "minf $(grep_count '(^|[[:space:]])minf([[:space:]]|$)' "$file")"

    echo "for $(grep_count '^[[:space:]]*(%[A-Za-z0-9_]+ = )?for[[:space:]]' "$file")"
    echo "select $(grep_count '(^|[[:space:]])select([[:space:]]|$)' "$file")"
    echo "iota $(grep_count '(^|[[:space:]])iota([[:space:]]|$)' "$file")"
    echo "cmpi $(grep_count '(^|[[:space:]])cmpi([[:space:]]|$)' "$file")"
    echo "ftof $(grep_count '(^|[[:space:]])ftof([[:space:]]|$)' "$file")"
    echo "itof $(grep_count '(^|[[:space:]])itof([[:space:]]|$)' "$file")"
    echo "exti $(grep_count '(^|[[:space:]])exti([[:space:]]|$)' "$file")"

    echo "occupancy $(first_pcre 'occupancy = \K[0-9]+' "$file")"
    echo "num_cta $(first_pcre 'num_cta_in_cga = \K[0-9]+' "$file")"
}

declare -A REF_OPS
declare -A GEN_OPS

while read -r op count; do
    [ -n "${op:-}" ] || continue
    REF_OPS[$op]="${count:-0}"
done < <(count_ops "$REF")

while read -r op count; do
    [ -n "${op:-}" ] || continue
    GEN_OPS[$op]="${count:-0}"
done < <(count_ops "$GEN")

CRITICAL_OPS=(load_ptr_tko store_ptr_tko load_view_tko store_view_tko reduce exp exp2 rsqrt fma mma for)
ARITH_OPS=(addf subf mulf divf maxf minf)
INFO_OPS=(select iota cmpi ftof itof exti)
HINT_OPS=(occupancy num_cta)

echo "================================================================"
echo "IR Op Comparison"
echo "================================================================"
printf "%-20s %8s %8s %8s %s\n" "Op" "Ref" "Gen" "Delta" "Verdict"
echo "----------------------------------------------------------------"

for op in "${CRITICAL_OPS[@]}" "${ARITH_OPS[@]}" "${INFO_OPS[@]}" "${HINT_OPS[@]}"; do
    ref=${REF_OPS[$op]:-0}
    gen=${GEN_OPS[$op]:-0}
    delta=$((gen - ref))
    verdict="MATCH"

    if [ "$delta" -ne 0 ]; then
        if in_array "$op" "${CRITICAL_OPS[@]}"; then
            verdict="CRITICAL_DIFF"
            append_critical "${op} ref=${ref} gen=${gen} delta=${delta}"
        elif in_array "$op" "${HINT_OPS[@]}"; then
            if [ "$ref" -ne 0 ] && [ "$gen" -ne 0 ] && [ "$ref" -ne "$gen" ]; then
                verdict="CRITICAL_DIFF"
                append_critical "${op} hint mismatch ref=${ref} gen=${gen}"
            elif [ "$ref" -ne 0 ] && [ "$gen" -eq 0 ]; then
                verdict="MISSING_HINT"
                append_warn "${op} hint missing in generated"
            else
                verdict="OK"
            fi
        elif in_array "$op" "${ARITH_OPS[@]}"; then
            abs_delta=${delta#-}
            if [ "$abs_delta" -le 2 ]; then
                verdict="WORKAROUND(${delta})"
                workaround_count=$((workaround_count + abs_delta))
            else
                verdict="CRITICAL_DIFF"
                append_critical "${op} ref=${ref} gen=${gen} delta=${delta} exceeds workaround allowance"
            fi
        else
            verdict="INFO(${delta})"
        fi
    fi

    printf "%-20s %8d %8d %+8d %s\n" "$op" "$ref" "$gen" "$delta" "$verdict"
done

echo ""
echo "Tile shapes:"
ref_shapes=$(join_egrep 'tile<[0-9]+x[A-Za-z0-9_]+>' "$REF")
gen_shapes=$(join_egrep 'tile<[0-9]+x[A-Za-z0-9_]+>' "$GEN")
echo "  Ref: ${ref_shapes:-none}"
echo "  Gen: ${gen_shapes:-none}"

if [ -n "$ref_shapes" ] && [ -n "$gen_shapes" ] && [ "$ref_shapes" != "$gen_shapes" ]; then
    append_critical "tile shape set differs"
fi

echo ""
echo "Reduce identities:"
ref_identities=$(grep -oE 'identities=\[[^]]+\]' "$REF" 2>/dev/null | sort -u)
gen_identities=$(grep -oE 'identities=\[[^]]+\]' "$GEN" 2>/dev/null | sort -u)
if [ -n "$ref_identities" ]; then
    echo "$ref_identities" | sed 's/^/  Ref: /'
else
    echo "  Ref: none"
fi
if [ -n "$gen_identities" ]; then
    echo "$gen_identities" | sed 's/^/  Gen: /'
else
    echo "  Gen: none"
fi
if [ "$ref_identities" != "$gen_identities" ]; then
    append_critical "reduce identities differ"
fi

echo ""
ref_rounding=$(grep_count_fixed "rounding_mode" "$REF")
gen_rounding=$(grep_count_fixed "rounding_mode" "$GEN")
echo "rounding_mode: ref=${ref_rounding} gen=${gen_rounding}"
if [ "$gen_rounding" -gt "$ref_rounding" ]; then
    append_warn "generated has extra rounding_mode attrs"
fi

echo ""
echo "Block mapping pattern:"
ref_has_swizzle=false
gen_has_swizzle=false
if grep -qE 'divi.*blockId|remi.*blockId' "$REF" 2>/dev/null; then ref_has_swizzle=true; fi
if grep -qE 'divi.*blockId|remi.*blockId' "$GEN" 2>/dev/null; then gen_has_swizzle=true; fi
ref_divi=$(grep_count '(^|[[:space:]])divi([[:space:]]|$)' "$REF")
gen_divi=$(grep_count '(^|[[:space:]])divi([[:space:]]|$)' "$GEN")
ref_remi=$(grep_count '(^|[[:space:]])remi([[:space:]]|$)' "$REF")
gen_remi=$(grep_count '(^|[[:space:]])remi([[:space:]]|$)' "$GEN")
echo "  Ref: swizzle=${ref_has_swizzle} divi=${ref_divi} remi=${ref_remi}"
echo "  Gen: swizzle=${gen_has_swizzle} divi=${gen_divi} remi=${gen_remi}"
if [ "$ref_has_swizzle" = true ] && [ "$gen_has_swizzle" = false ]; then
    append_warn "missing block swizzle; possible L2 locality regression"
fi

echo ""
echo "================================================================"
echo "Op Attribute Comparison"
echo "================================================================"

echo ""
echo "assume_div_by<N>:"
echo "  Reference:"
ref_assume_lines=$(grep -n 'assume div_by' "$REF" 2>/dev/null)
if [ -n "$ref_assume_lines" ]; then
    echo "$ref_assume_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi
echo "  Generated:"
gen_assume_lines=$(grep -n 'assume div_by' "$GEN" 2>/dev/null)
if [ -n "$gen_assume_lines" ]; then
    echo "$gen_assume_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi

ref_assume_ptr=$(grep_count 'assume div_by.*ptr' "$REF")
gen_assume_ptr=$(grep_count 'assume div_by.*ptr' "$GEN")
ref_assume_scalar=$(grep_count 'assume div_by.*tile<i(32|64)>' "$REF")
gen_assume_scalar=$(grep_count 'assume div_by.*tile<i(32|64)>' "$GEN")
gen_raw_ptr_ops=$(( ${GEN_OPS[load_ptr_tko]:-0} + ${GEN_OPS[store_ptr_tko]:-0} ))

echo "  Summary: Ref ptr=${ref_assume_ptr} scalar=${ref_assume_scalar} | Gen ptr=${gen_assume_ptr} scalar=${gen_assume_scalar}"

if [ "$ref_assume_ptr" -gt "$gen_assume_ptr" ]; then
    if [ "$gen_raw_ptr_ops" -gt 0 ]; then
        echo "  >> PERF_CRITICAL: generated raw-pointer path is missing pointer assume_div_by."
        append_critical "missing pointer assume_div_by coverage on raw-pointer generated path ref=${ref_assume_ptr} gen=${gen_assume_ptr}"
    else
        echo "  >> PERF_WARN: generated view/TMA path has fewer pointer assume_div_by ops."
        echo "     Policy: this is non-blocking unless a measured perf regression points here."
        append_warn "missing pointer assume_div_by coverage ref=${ref_assume_ptr} gen=${gen_assume_ptr}; generated has no raw pointer load/store ops"
    fi
fi

if [ "$gen_assume_scalar" -gt 0 ]; then
    echo "  >> CRITICAL: generated scalar assume_div_by is deprecated and can corrupt irregular shapes."
    append_critical "generated scalar assume_div_by present; remove scalar assume hints"
fi

echo ""
echo "Per-op attributes (flush_to_zero, rounding):"
echo "  Reference ops with attributes:"
ref_attr_lines=$(grep -nE '(addf|subf|mulf|divf|exp2?|negf).*flush_to_zero|rounding' "$REF" 2>/dev/null)
if [ -n "$ref_attr_lines" ]; then
    echo "$ref_attr_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi
echo "  Generated ops with attributes:"
gen_attr_lines=$(grep -nE '(addf|subf|mulf|divf|exp2?|negf).*flush_to_zero|rounding' "$GEN" 2>/dev/null)
if [ -n "$gen_attr_lines" ]; then
    echo "$gen_attr_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi

for op_type in addf subf mulf divf exp exp2 negf; do
    ref_ftz_op=$(grep_count "${op_type}.*flush_to_zero" "$REF")
    gen_ftz_op=$(grep_count "${op_type}.*flush_to_zero" "$GEN")
    ref_approx_op=$(grep_count "${op_type}.*rounding<approx>" "$REF")
    gen_approx_op=$(grep_count "${op_type}.*rounding<approx>" "$GEN")

    if [ "$ref_ftz_op" -ne "$gen_ftz_op" ] || [ "$ref_approx_op" -ne "$gen_approx_op" ]; then
        echo "  ${op_type}: FTZ ref=${ref_ftz_op} gen=${gen_ftz_op} | approx ref=${ref_approx_op} gen=${gen_approx_op}"
        if [ "$ref_ftz_op" -gt "$gen_ftz_op" ]; then
            append_critical "${op_type} missing flush_to_zero ref=${ref_ftz_op} gen=${gen_ftz_op}"
        fi
        if [ "$ref_approx_op" -gt "$gen_approx_op" ]; then
            append_critical "${op_type} missing rounding<approx> ref=${ref_approx_op} gen=${gen_approx_op}"
        fi
    fi
done

echo ""
echo "Load/store optimization hints:"
echo "  Reference:"
ref_hint_lines=$(grep -n 'optimization_hints.*=' "$REF" 2>/dev/null | grep -v 'entry @')
if [ -n "$ref_hint_lines" ]; then
    echo "$ref_hint_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi
echo "  Generated:"
gen_hint_lines=$(grep -n 'optimization_hints.*=' "$GEN" 2>/dev/null | grep -v 'entry @')
if [ -n "$gen_hint_lines" ]; then
    echo "$gen_hint_lines" | sed 's/^/    /'
else
    echo "    (none)"
fi

ref_allow_tma_true=$(grep_count 'allow_tma *= *true' "$REF")
gen_allow_tma_true=$(grep_count 'allow_tma *= *true' "$GEN")
ref_allow_tma_false=$(grep_count 'allow_tma *= *false' "$REF")
gen_allow_tma_false=$(grep_count 'allow_tma *= *false' "$GEN")
ref_latency=$(grep_count 'latency *=' "$REF")
gen_latency=$(grep_count 'latency *=' "$GEN")

echo "  Summary: Ref allow_tma_true=${ref_allow_tma_true} allow_tma_false=${ref_allow_tma_false} latency=${ref_latency} | Gen allow_tma_true=${gen_allow_tma_true} allow_tma_false=${gen_allow_tma_false} latency=${gen_latency}"

if [ "$ref_allow_tma_true" -gt "$gen_allow_tma_true" ]; then
    append_critical "allow_tma=true count mismatch ref=${ref_allow_tma_true} gen=${gen_allow_tma_true}"
fi
if [ "$ref_allow_tma_false" -ne "$gen_allow_tma_false" ]; then
    append_warn "allow_tma=false count mismatch ref=${ref_allow_tma_false} gen=${gen_allow_tma_false}"
fi
if [ "$ref_latency" -gt "$gen_latency" ]; then
    append_warn "latency hint count mismatch ref=${ref_latency} gen=${gen_latency}"
elif [ "$gen_latency" -gt "$ref_latency" ]; then
    append_warn "generated extra latency hints ref=${ref_latency} gen=${gen_latency}"
fi

echo ""
echo "MMA accumulator types:"
ref_mma_acc=$(join_pcre 'mmaf?.*tile<[0-9]+x[0-9]+x\w+>' "$REF" | tr ' ' '\n' | grep -oP 'tile<[0-9]+x[0-9]+x\w+>$' 2>/dev/null | sort -u | tr '\n' ' ')
gen_mma_acc=$(join_pcre 'mmaf?.*tile<[0-9]+x[0-9]+x\w+>' "$GEN" | tr ' ' '\n' | grep -oP 'tile<[0-9]+x[0-9]+x\w+>$' 2>/dev/null | sort -u | tr '\n' ' ')
echo "  Ref: ${ref_mma_acc:-none}"
echo "  Gen: ${gen_mma_acc:-none}"
if [ -n "$ref_mma_acc" ] && [ -n "$gen_mma_acc" ] && [ "$ref_mma_acc" != "$gen_mma_acc" ]; then
    append_critical "MMA accumulator ref=${ref_mma_acc} gen=${gen_mma_acc}"
fi

echo ""
ref_has_bf16=$(grep_count_fixed 'bf16' "$REF")
gen_has_bf16=$(grep_count_fixed 'bf16' "$GEN")
ref_has_f16=$(grep_count_fixed 'f16' "$REF")
gen_has_f16=$(grep_count_fixed 'f16' "$GEN")
echo "dtype usage: ref(f16=${ref_has_f16}, bf16=${ref_has_bf16}) gen(f16=${gen_has_f16}, bf16=${gen_has_bf16})"
if [ "$ref_has_bf16" -gt 0 ] && [ "$gen_has_bf16" -eq 0 ]; then
    append_warn "reference uses bf16 but generated does not"
fi

echo ""
echo "Tensor view layout attributes:"
ref_layout=$(grep -oE 'strides=\[[^]]+\]|dim_map=\[[^]]+\]|tile=\([^)]+\)' "$REF" 2>/dev/null | sort | uniq -c)
gen_layout=$(grep -oE 'strides=\[[^]]+\]|dim_map=\[[^]]+\]|tile=\([^)]+\)' "$GEN" 2>/dev/null | sort | uniq -c)
echo "  Reference:"
if [ -n "$ref_layout" ]; then
    echo "$ref_layout" | sed 's/^/    /'
else
    echo "    (none)"
fi
echo "  Generated:"
if [ -n "$gen_layout" ]; then
    echo "$gen_layout" | sed 's/^/    /'
else
    echo "    (none)"
fi
if [ "$ref_layout" != "$gen_layout" ]; then
    append_warn "layout attribute multiset differs; inspect dim_map/strides/tile shapes"
fi

echo ""
echo "================================================================"
if [ "$critical_fail" -gt 0 ]; then
    echo "VERDICT: FAIL_FIXABLE (${critical_fail} critical diff(s))"
    [ -n "$notes" ] && printf "%b\n" "$notes"
    exit 1
elif [ "$workaround_count" -gt 0 ] || [ "$note_count" -gt 0 ]; then
    echo "VERDICT: PASS_WITH_NOTES (${workaround_count} workaround op(s), ${note_count} note(s))"
    [ -n "$notes" ] && printf "%b\n" "$notes"
    exit 2
else
    echo "VERDICT: PASS"
    exit 0
fi
