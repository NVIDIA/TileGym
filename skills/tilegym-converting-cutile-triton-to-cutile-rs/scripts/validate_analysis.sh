#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate analysis.json has all required fields including correctness_reference and correctness_tolerance.
# Called by Agent A (post-step validation) and validate_kernel.sh (final gate).
#
# Usage: bash validate_analysis.sh <kernel_name>
# Exit 0 = valid, Exit 1 = missing required fields

set -euo pipefail

KERNEL_NAME="${1:?Usage: validate_analysis.sh <kernel_name>}"

# Auto-resolve CUTILE_RS_ROOT from this script's location
: "${CUTILE_RS_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

# CUTILE_KERNEL_OUT_ROOT locates the per-kernel working dir; without it (set -u)
# the ANALYSIS expansion below aborts with a cryptic "unbound variable" — guard
# it so the failure names the missing var.
: "${CUTILE_KERNEL_OUT_ROOT:?CUTILE_KERNEL_OUT_ROOT not set — cannot locate reference/analysis.json}"

ANALYSIS="${CUTILE_KERNEL_OUT_ROOT}/${KERNEL_NAME}/reference/analysis.json"

[ -f "$ANALYSIS" ] || { echo "FAIL: $ANALYSIS not found"; exit 1; }
[ -s "$ANALYSIS" ] || { echo "FAIL: $ANALYSIS is empty"; exit 1; }

# Validate JSON is parseable
if ! python3 -c "import json; json.load(open('${ANALYSIS}'))" 2>/dev/null; then
    echo "FAIL: $ANALYSIS is not valid JSON"
    exit 1
fi

# Check required fields using python (pass path via env var)
ANALYSIS_PATH="$ANALYSIS" python3 -c "
import json, os, sys

d = json.load(open(os.environ['ANALYSIS_PATH']))

errors = []
warnings = []

# --- Required top-level fields ---
for f in ['kernel_name', 'source', 'launch_path', 'pattern', 'ops_used', 'test_perf_configs']:
    if f not in d:
        errors.append(f'Missing required field: {f}')

# --- Reference backend selection ---
ref_backend = d.get('reference_backend')
if ref_backend is None:
    warnings.append('Missing reference_backend — should be \"cutile\" or \"triton\" (defaults to cutile for backward compat)')
elif ref_backend not in ('cutile', 'triton'):
    errors.append(f'reference_backend must be \"cutile\" or \"triton\", got \"{ref_backend}\"')

ref_selection = d.get('reference_backend_selection')
if ref_selection is None:
    warnings.append('Missing reference_backend_selection — Agent A should compare cutile vs triton perf and pick the winner')
elif isinstance(ref_selection, dict):
    if 'winner' not in ref_selection:
        warnings.append('reference_backend_selection missing \"winner\" field')
    if 'cutile_geo_mean_ms' not in ref_selection and 'per_variant' not in ref_selection:
        warnings.append('reference_backend_selection has no perf data (need cutile_geo_mean_ms or per_variant)')

# --- test_perf_configs must have shapes ---
configs = d.get('test_perf_configs', {})
if isinstance(configs, dict):
    total_shapes = sum(len(v.get('shapes', [])) for v in configs.values())
elif isinstance(configs, list):
    total_shapes = sum(len(c.get('param_list', [])) for c in configs)
else:
    total_shapes = 0
if total_shapes == 0:
    errors.append('test_perf_configs has 0 shapes')

# --- kernel_variants (multi-variant ops) ---
variants = d.get('kernel_variants')
if variants is not None:
    if not isinstance(variants, list):
        errors.append('kernel_variants must be a list of variant dicts')
    elif len(variants) == 0:
        warnings.append('kernel_variants is empty — should have at least 1 entry')
    else:
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                errors.append(f'kernel_variants[{i}] must be a dict')
                continue
            vname = v.get('variant_name', f'variant_{i}')
            for req_key in ['variant_name', 'kernel_function', 'reference_ir']:
                if req_key not in v:
                    errors.append(f'kernel_variants[{i}] ({vname}) missing {req_key}')
            if 'dispatch_condition' not in v:
                warnings.append(f'kernel_variants[{i}] ({vname}) missing dispatch_condition')

            # Per-variant reference_backend (pipeline rule 15: dual-backend per-variant selection)
            v_ref_backend = v.get('reference_backend')
            if v_ref_backend is None:
                warnings.append(f'kernel_variants[{i}] ({vname}) missing reference_backend — should be \"cutile\" or \"triton\"')
            elif v_ref_backend not in ('cutile', 'triton'):
                errors.append(f'kernel_variants[{i}] ({vname}) reference_backend must be \"cutile\" or \"triton\", got \"{v_ref_backend}\"')

            # Per-variant reference_ir file must exist on disk
            v_ir = v.get('reference_ir')
            if v_ir:
                import os as _os
                ir_path = _os.path.join(_os.path.dirname(os.environ['ANALYSIS_PATH']), '..', v_ir)
                ir_path_abs = _os.path.normpath(ir_path)
                if not _os.path.isfile(ir_path_abs):
                    errors.append(f'kernel_variants[{i}] ({vname}) reference_ir file not found: {v_ir} (resolved: {ir_path_abs})')

        # Per-variant perf data in reference_backend_selection
        if ref_selection and isinstance(ref_selection, dict):
            pv_sel = ref_selection.get('per_variant')
            if pv_sel is None:
                warnings.append('reference_backend_selection missing per_variant — should have per-variant perf comparison')
            elif isinstance(pv_sel, dict):
                for i, v in enumerate(variants):
                    if not isinstance(v, dict):
                        continue
                    vname = v.get('variant_name', f'variant_{i}')
                    if vname not in pv_sel:
                        warnings.append(f'reference_backend_selection.per_variant missing entry for \"{vname}\"')

        # If variants exist, config_to_variant should map each test config to a variant
        ctv = d.get('config_to_variant')
        if ctv is None:
            warnings.append('kernel_variants exist but config_to_variant is missing — Agent E needs this to compare variant-to-variant')
        elif not isinstance(ctv, dict) or len(ctv) == 0:
            warnings.append('config_to_variant is empty — should map each test_perf config to its variant')

# --- correctness_reference (pipeline rule 8) ---
ref = d.get('correctness_reference')
if ref is None:
    errors.append('Missing correctness_reference — extract def reference(...) from tilegym test class')
elif not isinstance(ref, dict):
    errors.append('correctness_reference must be a dict with function and source keys')
else:
    if not ref.get('function'):
        errors.append('correctness_reference.function is empty')
    if 'source' not in ref:
        warnings.append('correctness_reference missing source (file:line)')

# --- correctness_tolerance (pipeline rule 9) ---
tol = d.get('correctness_tolerance')
dtype_entries = {}
if tol is None:
    errors.append('Missing correctness_tolerance — extract atol/rtol from tilegym test_op')
elif not isinstance(tol, dict):
    errors.append('correctness_tolerance must be a dict with dtype keys')
else:
    dtype_entries = {k: v for k, v in tol.items() if k not in ('source', 'notes')}
    if len(dtype_entries) == 0:
        errors.append('correctness_tolerance has no dtype entries (need f32/f16/bf16)')
    for dk, dv in dtype_entries.items():
        if not isinstance(dv, dict):
            errors.append(f'correctness_tolerance[{dk}] must have atol and rtol')
        else:
            if 'atol' not in dv:
                errors.append(f'correctness_tolerance[{dk}] missing atol')
            if 'rtol' not in dv:
                errors.append(f'correctness_tolerance[{dk}] missing rtol')
    if 'source' not in tol:
        warnings.append('correctness_tolerance missing source (file:line)')

# --- Autotune fields (if launch_path is autotune or autotune_backends present) ---
launch_path = d.get('launch_path', 'direct')
at_backends = d.get('autotune_backends', [])
at_configs = d.get('autotune_configs', [])

if launch_path == 'autotune' or len(at_backends) > 0:
    # Check top-level autotune_configs OR per-variant autotune_configs
    has_any_autotune_configs = False

    # Check per-variant autotune_configs
    if variants and isinstance(variants, list):
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                continue
            v_at = v.get('autotune_configs')
            if v_at is not None and isinstance(v_at, list) and len(v_at) > 0:
                has_any_autotune_configs = True
                for j, cfg in enumerate(v_at):
                    if not isinstance(cfg, dict):
                        errors.append(f'kernel_variants[{i}].autotune_configs[{j}] must be a dict')
                    elif not any(k.startswith('BLOCK_') or k.startswith('TILE_') for k in cfg):
                        warnings.append(f'kernel_variants[{i}].autotune_configs[{j}] has no BLOCK_*/TILE_* fields')

    # Check top-level autotune_configs (for single-variant or union)
    if at_configs and isinstance(at_configs, list) and len(at_configs) > 0:
        has_any_autotune_configs = True
        for i, cfg in enumerate(at_configs):
            if not isinstance(cfg, dict):
                errors.append(f'autotune_configs[{i}] must be a dict')
            elif not any(k.startswith('BLOCK_') or k.startswith('TILE_') for k in cfg):
                warnings.append(f'autotune_configs[{i}] has no BLOCK_*/TILE_* fields: {list(cfg.keys())}')

    if not has_any_autotune_configs:
        errors.append('autotune detected (launch_path=autotune or autotune_backends set) but no autotune_configs found. '
                       'Must be in top-level autotune_configs OR kernel_variants[].autotune_configs.')

    if not at_backends or not isinstance(at_backends, list):
        warnings.append('autotune_backends missing — should list which backends use autotune (e.g. [\"cutile\", \"triton\"])')
    at_key = d.get('autotune_key')
    if not at_key:
        warnings.append('autotune_key missing — should list cache key dimensions (e.g. [\"M\", \"N\", \"K\"])')

n_at_configs = len(at_configs) if isinstance(at_configs, list) else 0
# Count per-variant configs too
n_variant_at_configs = 0
if variants and isinstance(variants, list):
    for v in variants:
        if isinstance(v, dict):
            v_at = v.get('autotune_configs')
            if v_at and isinstance(v_at, list):
                n_variant_at_configs += len(v_at)

# --- Output ---
for e in errors:
    print(f'ERROR: {e}')
for w in warnings:
    print(f'WARN: {w}')

ref_fn = ref.get('function', 'MISSING')[:60] if isinstance(ref, dict) else 'MISSING'
tol_list = list(dtype_entries.keys())
n_variants = len(variants) if variants and isinstance(variants, list) else 0
variant_names = [v.get('variant_name','?') for v in variants] if variants and isinstance(variants, list) else []
print(f'---')
print(f'kernel: {d.get(\"kernel_name\", \"MISSING\")}')
print(f'shapes: {total_shapes}')
print(f'variants: {n_variants} {variant_names}')
print(f'reference: {ref_fn}')
print(f'tolerance: {tol_list}')
print(f'autotune: {\"yes\" if (launch_path == \"autotune\" or at_backends) else \"no\"} backends={at_backends} top_configs={n_at_configs} variant_configs={n_variant_at_configs}')
print(f'errors: {len(errors)}, warnings: {len(warnings)}')
sys.exit(1 if errors else 0)
"
rc=$?

if [ $rc -ne 0 ]; then
    echo "FAIL: analysis.json validation failed. Re-run Agent A."
    exit 1
else
    echo "PASS: analysis.json has all required fields"
    exit 0
fi
