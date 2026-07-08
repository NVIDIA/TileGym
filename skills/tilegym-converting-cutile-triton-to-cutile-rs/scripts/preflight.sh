#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

# Hard gate: refuses to proceed with the tilegym-converting-cutile-triton-to-cutile-rs skill until
#   1 git-tracked repo (TILEGYM_PATH — holds tests, Python wrappers, AND the
#     cutile_rs Rust under src/tilegym/ops/cutile_rs/) and
#   4 cuTile toolchain paths (TILEIRAS_BIN, CUDA_TILE_OPT_BIN,
#   TRITON_TILEIR_PYTHONPATH, CUDA_TOOLKIT_PATH) are set and verified on disk.
# There is NO separate cutile-rs checkout and NO CUTILE_RS_ROOT anymore — the
# aggregated cutile_kernels crate (crates.io PINNED deps, one libcutile_kernels.so)
# lives inside the tilegym tree.
# Prints a STOP banner with the exact next action when the gate fails;
# the banner ends up in the orchestrator context, making the gate
# behaviorally enforceable rather than text-only.
#
# Required env vars (5 total):
#   export TILEGYM_PATH=/abs/path/to/tilegym
#   export TILEIRAS_BIN=/abs/path/to/tileiras
#   export CUDA_TILE_OPT_BIN=/abs/path/to/cuda-tile-opt
#   export TRITON_TILEIR_PYTHONPATH=/abs/path/to/Triton-TileIR/python
#   export CUDA_TOOLKIT_PATH=/abs/path/to/cuda-toolkit
#
# Optional (defaults shown): the loader autobuilds the cutile_kernels crate on
# first use; override with:
#   export CUTILE_RS_AUTOBUILD=1                    # on by default; 0 = use prebuilt
#   export CUTILE_RS_KERNELS_DIR=$TILEGYM_PATH/src/tilegym/ops/cutile_rs/cutile_kernels
#
# Usage:
#   bash scripts/preflight.sh
#
# Returns:
#   0 = paths set + verified → safe to read the rest of SKILL.md
#   1 = paths missing/invalid → AskUserQuestion + re-export + re-run

set -uo pipefail

red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; bld=$'\033[1m'; rst=$'\033[0m'

banner_stop() {
  printf '%s' "${red}${bld}"
  echo "==============================================================="
  echo "  STOP -- tilegym-converting-cutile-triton-to-cutile-rs preflight FAILED"
  echo "==============================================================="
  printf '%s' "${rst}"
}

banner_ok() {
  printf '%s' "${grn}${bld}"
  echo "==============================================================="
  echo "  OK -- tilegym-converting-cutile-triton-to-cutile-rs preflight PASSED"
  echo "==============================================================="
  printf '%s' "${rst}"
}

check_dir() {
  local name="$1" val="${2:-}" desc="$3"
  if [ -z "$val" ]; then
    echo "  ${red}MISSING${rst}    $name        ($desc)"
    return 1
  fi
  if [ ! -d "$val" ]; then
    echo "  ${red}NOT A DIR${rst}  $name=$val"
    echo "             ($desc)"
    return 1
  fi
  echo "  ${grn}OK${rst}         $name=$val"
  return 0
}

check_executable() {
  local name="$1" val="${2:-}" desc="$3"
  if [ -z "$val" ]; then
    echo "  ${red}MISSING${rst}    $name        ($desc)"
    return 1
  fi
  if [ ! -x "$val" ]; then
    echo "  ${red}NOT EXECUTABLE${rst}  $name=$val"
    echo "                  ($desc)"
    return 1
  fi
  echo "  ${grn}OK${rst}         $name=$val"
  return 0
}

fail=0
echo
echo "Gate — paths (1 git repo + 4 toolchain paths):"
echo
echo "  ${bld}Git repo${rst}:"
check_dir "TILEGYM_PATH"   "${TILEGYM_PATH:-}"   "tilegym checkout (tests + Python wrappers + cutile_rs Rust under src/tilegym/ops/cutile_rs/)" || fail=1
echo
echo "  ${bld}Toolchain${rst} (existence-checked):"
check_executable "TILEIRAS_BIN"      "${TILEIRAS_BIN:-}"      "cuTile compiler binary (tileiras)" || fail=1
check_executable "CUDA_TILE_OPT_BIN" "${CUDA_TILE_OPT_BIN:-}" "MLIR canonicalizer (cuda-tile-opt) — used by Agent B for IR diff" || fail=1
check_dir        "TRITON_TILEIR_PYTHONPATH" "${TRITON_TILEIR_PYTHONPATH:-}" "Triton-TileIR Python bindings dir (TileIR backend) — required for Triton-TileIR IR dump + Triton-TileIR correctness tests" || fail=1
check_dir        "CUDA_TOOLKIT_PATH" "${CUDA_TOOLKIT_PATH:-}" "CUDA toolkit root (cuda-bindings/build.rs requires this; e.g. /usr/local/cuda)" || fail=1
echo
echo "  ${bld}Per-kernel working dir${rst} (Agent A/B/C/D write here; eval harness depends on it):"
if [ -z "${CUTILE_KERNEL_OUT_ROOT:-}" ]; then
  echo "  ${red}MISSING${rst}    CUTILE_KERNEL_OUT_ROOT  (export to /workspace/cutile_kernel_out or similar)"
  fail=1
else
  mkdir -p "$CUTILE_KERNEL_OUT_ROOT" 2>/dev/null || true
  check_dir "CUTILE_KERNEL_OUT_ROOT" "$CUTILE_KERNEL_OUT_ROOT" "Per-kernel output root for agent artifacts (kernel.rs / ffi.rs / reference/ / generated/ / reports/). Kernel Rust is wired into the aggregated cutile_kernels crate under tilegym; there is no per-kernel Cargo.toml." || fail=1
fi
echo

if [ "$fail" -ne 0 ]; then
  banner_stop
  echo
  echo "${bld}MANDATORY NEXT ACTION${rst} (orchestrator):"
  echo
  echo "  1. Call ${bld}AskUserQuestion${rst} with one question per missing path."
  echo "     Do NOT batch-guess. Do NOT default from cwd / git / memory."
  echo
  echo "  2. After the user answers, echo all paths back so they can correct typos."
  echo
  echo "  3. Export them in the shell that will run subsequent tool calls:"
  echo "       export TILEGYM_PATH=/abs/path/to/tilegym"
  echo "       export TILEIRAS_BIN=/abs/path/to/tileiras"
  echo "       export CUDA_TILE_OPT_BIN=/abs/path/to/cuda-tile-opt"
  echo "       export TRITON_TILEIR_PYTHONPATH=/abs/path/to/Triton-TileIR/python"
  echo "       export CUDA_TOOLKIT_PATH=/abs/path/to/cuda-toolkit"
  echo
  echo "  4. Re-run THIS script. It must exit 0 before any other tool call:"
  echo "       bash ${BASH_SOURCE[0]}"
  echo
  echo "${ylw}HARD RULES${rst} (violations = task failure):"
  echo "  - Do NOT call Read / Edit / Write / Bash / Agent for skill work"
  echo "    until this script exits 0."
  echo "  - Do NOT 'just look at the test file' to estimate scope --"
  echo "    that is the 'minimal edit' anti-pattern explicitly banned by"
  echo "    SKILL.md line 30."
  echo "  - Do NOT assume the cwd repo is TILEGYM_PATH. Ask the user."
  echo
  banner_stop
  exit 1
fi

echo "  ${grn}Gate passed${rst} — all 5 paths verified."
echo "    TILEGYM_PATH         = $TILEGYM_PATH"
echo "    TILEIRAS_BIN         = $TILEIRAS_BIN"
echo "    CUDA_TILE_OPT_BIN    = $CUDA_TILE_OPT_BIN"
echo "    TRITON_TILEIR_PYTHONPATH = $TRITON_TILEIR_PYTHONPATH"
echo "    CUDA_TOOLKIT_PATH    = $CUDA_TOOLKIT_PATH"
echo

banner_ok
echo
echo "  All gates passed. You may now read the rest of SKILL.md and spawn Agent A."
echo
exit 0
