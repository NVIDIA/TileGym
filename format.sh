#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Quick formatting script for TileGym development.
# Runs the pre-commit hooks, adds SPDX headers, and formats/sorts with ruff.

set -e

RUFF_VERSION="0.14.9"

echo "🔍 Checking pre-commit installation..."
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

echo ""
echo "Installing pre-commit hooks..."
pre-commit install-hooks

echo ""
echo "Running all pre-commit hooks..."
# Run pre-commit on all files; continue even if some hooks fail
# (several hooks are auto-fixers and will modify files).
set +e
pre-commit run --all-files
set -e

echo ""
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
echo "✅ Done! Pre-commit hooks ran, SPDX headers added, code formatted, imports sorted."
