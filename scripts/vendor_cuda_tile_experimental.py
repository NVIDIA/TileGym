#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
Vendor cuda-tile-experimental into src/cuda/tile_experimental for wheel builds.
Run before 'python -m build --wheel' so the wheel is self-contained.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TARGET = REPO_ROOT / "src" / "cuda" / "tile_experimental"
GIT_URL = "git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"


def main() -> int:
    print("Vendoring cuda-tile-experimental into wheel...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"cuda-tile-experimental @ {GIT_URL}"],
        check=True,
    )
    # Get installed package path
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cuda.tile_experimental as m; print(m.__path__[0])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    src = Path(result.stdout.strip())
    VENDOR_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if VENDOR_TARGET.exists():
        shutil.rmtree(VENDOR_TARGET)
    shutil.copytree(src, VENDOR_TARGET)
    print(f"Copied {src} -> {VENDOR_TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
