#!/bin/bash
# Quick formatting script for TileGym development
# Formats code and sorts imports using ruff

set -e

RUFF_VERSION="0.14.9"

echo "🔍 Checking ruff installation..."
if ! python3 -m ruff --version 2>/dev/null | grep -q "$RUFF_VERSION"; then
    echo "📦 Installing ruff $RUFF_VERSION..."
    pip install "ruff==$RUFF_VERSION"
fi

echo ""
echo "📋 Sorting imports..."
python3 -m ruff check --select I --fix .

echo ""
echo "✨ Formatting code..."
python3 -m ruff format .

echo ""
echo "✅ Done! Code is formatted and imports are sorted."

