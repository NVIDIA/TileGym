<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Original paged-KV decode example

Start your optimization experiment with `decode_cutile.py` in this directory. Keep the correctness tests and benchmark unchanged. Follow [the parent README](../README.md) to prepare a workspace, remove the supplied `solution/` before your own agent run, collect baseline/final JSON data, or reproduce the supplied solution separately.

Run `python -m pytest test_decode.py` and `python bench_decode.py --json baseline.json` from this directory to validate and measure the original kernel. Use a separate JSON output file for the optimized run.
