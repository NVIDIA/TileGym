#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Validate a dumped CUDA Tile IR file.
# Usage: bash validate_ir.sh <ir_file> <expected_kernel_name>
# Exit 0 = valid, Exit 1 = invalid

set -euo pipefail

IR_FILE="${1:?Usage: validate_ir.sh <ir_file> <expected_kernel_name>}"
KERNEL_NAME="${2:?Usage: validate_ir.sh <ir_file> <expected_kernel_name>}"

[ -f "$IR_FILE" ] || { echo "FAIL: $IR_FILE not found"; exit 1; }
[ -s "$IR_FILE" ] || { echo "FAIL: $IR_FILE is empty"; exit 1; }

errors=0

# Check 1: Must have cuda_tile.module header (MLIR format, not Python IR)
if ! grep -q "cuda_tile.module" "$IR_FILE"; then
    echo "FAIL: Missing 'cuda_tile.module' — this is NOT valid MLIR format"
    echo "      (Got Python-level IR instead? Check CUDA_TILE_DUMP_TILEIR was used)"
    errors=$((errors + 1))
fi

# Check 2: Must have entry whose symbol contains the kernel name as a substring.
# Triton-TileIR @tile_jit produces mangled names like
#   @_static_persistent_bmm_kernel_Kt1_A3f16_...   (multi-variant)
#   @_bmm_kernel_memref_v2opt3_...                 (single-variant)
# cuTile-Python @ct.kernel produces literal `@bmm`.
# cutile-rs `#[cutile::module]` produces `@<kernel>_kernel_<hash>`.
# We accept any entry symbol that contains "${KERNEL_NAME}" as a substring
# (matches all three conventions without false-positives — the kernel name
# is op-specific enough that "bmm" inside another op's mangled name is
# extremely unlikely).
if ! grep -qE "entry @[a-zA-Z0-9_]*${KERNEL_NAME}[a-zA-Z0-9_]*" "$IR_FILE"; then
    echo "FAIL: No entry symbol contains '${KERNEL_NAME}' as substring"
    # Show what entries exist
    echo "      Found entries:"
    grep -o "entry @[a-zA-Z0-9_]*" "$IR_FILE" | sed 's/^/        /'
    errors=$((errors + 1))
fi

# Check 3: Must have optimization_hints (even if empty)
if ! grep -q "optimization_hints" "$IR_FILE"; then
    echo "WARN: Missing 'optimization_hints' in entry"
fi

# Check 4: Must not be Python-level IR (Tile[PointerTy...] format)
if grep -q "Tile\[PointerTy\|typed_const\|assume_bounded(x=" "$IR_FILE"; then
    echo "FAIL: File contains Python-level IR, not MLIR"
    echo "      Use CUDA_TILE_DUMP_TILEIR env var to get MLIR format"
    errors=$((errors + 1))
fi

# Check 5: Should have at least some ops
op_count=$(grep -cE "load_ptr_tko|store_ptr_tko|load_view_tko|store_view_tko|mmaf|reduce|make_partition_view" "$IR_FILE" || true)
if [ "$op_count" -eq 0 ]; then
    echo "WARN: No recognized ops found (load/store/mma/reduce/partition)"
fi

if [ $errors -gt 0 ]; then
    echo "---"
    echo "FAIL: ${errors} error(s) in IR validation"
    exit 1
else
    lines=$(wc -l < "$IR_FILE")
    # Show the matched entry symbol for transparency
    matched=$(grep -oE "entry @[a-zA-Z0-9_]*${KERNEL_NAME}[a-zA-Z0-9_]*" "$IR_FILE" | head -1)
    echo "PASS: Valid MLIR IR matched ${matched:-@${KERNEL_NAME}} (${lines} lines, ${op_count} key ops)"
    exit 0
fi
