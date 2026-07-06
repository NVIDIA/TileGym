#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Quick formatting script for TileGym development
# Formats code and sorts imports using ruff

set -e

RUFF_VERSION="0.14.9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Checking pre-commit installation..."
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

echo ""
echo "Installing pre-commit hooks..."
pre-commit install-hooks

echo ""
echo "Running all formatting and checks..."
# Run pre-commit on all files, continue even if some checks fail
# (some hooks like ruff are auto-fixers and will modify files)
set +e  # Temporarily allow errors to see full output
pre-commit run --all-files
exit_code=$?
set -e  # Re-enable error exit

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "Some files were modified by auto-fixers. Running again to verify..."
    pre-commit run --all-files
fi

echo ""
echo "✅ Done! All checks of pre-commit hooks completed."

echo "🔍 Checking ruff installation..."
if ! python3 -m ruff --version 2>/dev/null | grep -q "$RUFF_VERSION"; then
    echo "📦 Installing ruff $RUFF_VERSION..."
    pip install "ruff==$RUFF_VERSION"
fi

echo ""
echo "📝 Adding SPDX headers to files..."
python3 .github/scripts/check_spdx_headers.py --action write

echo ""
echo "📋 Sorting imports..."
python3 -m ruff check --select I --fix .

echo ""
echo "✨ Formatting code..."
python3 -m ruff format .

echo ""
echo "🦀 Formatting Rust (.rs) files..."
# rustfmt is optional: most contributors are Python-only and have no Rust
# toolchain. Skip (with a hint) when it is absent instead of force-installing.
# Run rustfmt per-file rather than `cargo fmt` so standalone .rs (cutile-rs
# skill examples, per-op kernel.rs/ffi.rs) are covered, not just crate members.
if command -v rustfmt >/dev/null 2>&1; then
    rs_files=$(git ls-files '*.rs')
    if [ -n "$rs_files" ]; then
        echo "$rs_files" | xargs rustfmt --edition 2024
        echo "✅ rustfmt formatted $(printf '%s\n' "$rs_files" | wc -l) .rs file(s)."
    else
        echo "No tracked .rs files; skipping rustfmt."
    fi
else
    echo "⚠️  rustfmt not found — skipping .rs formatting."
    echo "    Install with: rustup component add rustfmt   (see https://rustup.rs)"
fi

echo ""
echo "✅ Done! SPDX headers added, code is formatted, and imports are sorted."
