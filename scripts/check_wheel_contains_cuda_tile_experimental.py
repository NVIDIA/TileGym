#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
Check that the built wheel contains the vendored cuda.tile_experimental package.
Run after 'python -m build --wheel'. Exits 0 if present, 1 with message if not.
"""

import sys
import zipfile
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist"
REQUIRED_PREFIX = "cuda/tile_experimental/"


def main() -> int:
    wheels = list(DIST.glob("*.whl"))
    if not wheels:
        print("ERROR: No wheel found in dist/", file=sys.stderr)
        return 1
    if len(wheels) > 1:
        print("WARNING: Multiple wheels, checking first:", wheels[0].name, file=sys.stderr)
    whl = wheels[0]
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
    found = [n for n in names if n.startswith(REQUIRED_PREFIX)]
    if not found:
        print(
            f"ERROR: Wheel {whl.name} does not contain {REQUIRED_PREFIX!r} (vendored cuda-tile-experimental).",
            file=sys.stderr,
        )
        print("Vendor step may have failed or setup.py did not include the package.", file=sys.stderr)
        return 1
    print(f"OK: Wheel contains vendored cuda.tile_experimental ({len(found)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
