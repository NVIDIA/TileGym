#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
Vendor cuda-tile-experimental into src/cuda/tile_experimental for wheel builds.
Run before 'python -m build --wheel' so the wheel is self-contained.
"""

import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TARGET = REPO_ROOT / "src" / "cuda" / "tile_experimental"
GIT_URL = "git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"


def find_installed_path() -> Optional[Path]:
    """Locate cuda/tile_experimental in site-packages without importing (import may load CUDA)."""
    for sp in site.getsitepackages():
        # Preferred: namespace layout cuda/tile_experimental
        candidate = Path(sp) / "cuda" / "tile_experimental"
        if candidate.is_dir():
            return candidate
        # Fallback: flat layout cuda_tile_experimental (we'll copy into cuda/tile_experimental)
        flat = Path(sp) / "cuda_tile_experimental"
        if flat.is_dir():
            return flat
    return None


def main() -> int:
    print("Vendoring cuda-tile-experimental into wheel...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"cuda-tile-experimental @ {GIT_URL}"],
        check=True,
    )
    src = find_installed_path()
    if not src:
        print("ERROR: cuda/tile_experimental not found in site-packages after install.", file=sys.stderr)
        return 1
    VENDOR_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if VENDOR_TARGET.exists():
        shutil.rmtree(VENDOR_TARGET)
    shutil.copytree(src, VENDOR_TARGET)
    print(f"Copied {src} -> {VENDOR_TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
