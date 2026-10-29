# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for tests/benchmark/run_all_json.py.

The tables below are what triton.testing.perf_report prints with print_data=True
(a pandas DataFrame). Triton >= 3.6 adds the y-axis label to each measurement
column, so the header reads "CuTile (GB/s)" instead of "CuTile".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests" / "benchmark"))

from run_all_json import find_benchmark_files
from run_all_json import parse_benchmark_output
from run_all_json import parse_skip_reason

SOFTMAX_TRITON_3_5 = """\
softmax-performance-float32-GBps:
        N  CuTile  PyTorch
0  1024.0  1501.2   1203.4
1  2048.0  1622.9   1310.0
"""

SOFTMAX_TRITON_3_6 = """\
softmax-performance-float32-GBps:
        N  CuTile (GB/s)  PyTorch (GB/s)
0  1024.0         1501.2          1203.4
1  2048.0         1622.9          1310.0
"""

SOFTMAX_CONFIGS = [
    {"N": "1024.0", "CuTile": 1501.2, "PyTorch": 1203.4},
    {"N": "2048.0", "CuTile": 1622.9, "PyTorch": 1310.0},
]


def test_parses_triton_3_6_unit_columns():
    results = parse_benchmark_output(SOFTMAX_TRITON_3_6)

    assert results == [{"name": "softmax-performance-float32", "unit": "GBps", "configs": SOFTMAX_CONFIGS}]


def test_parses_columns_without_units():
    results = parse_benchmark_output(SOFTMAX_TRITON_3_5)

    assert results == [{"name": "softmax-performance-float32", "unit": "GBps", "configs": SOFTMAX_CONFIGS}]


def test_keeps_multi_word_line_names():
    output = """\
rmsnorm-performance-bfloat16-GBps:
        N  CuTile static persistent (GB/s)  PyTorch (GB/s)
0  1024.0                           980.50          610.25
1  8192.0                          1450.75          902.00
"""
    results = parse_benchmark_output(output)

    assert results[0]["configs"] == [
        {"N": "1024.0", "CuTile static persistent": 980.5, "PyTorch": 610.25},
        {"N": "8192.0", "CuTile static persistent": 1450.75, "PyTorch": 902.0},
    ]


def test_multiple_x_columns():
    output = """\
bmm-performance-float16-TFLOPS:
        M       N       K  CuTile (TFLOPS)  PyTorch (TFLOPS)
0   512.0   512.0   512.0             95.3              88.1
1  4096.0  4096.0  4096.0            612.4             590.8
"""
    results = parse_benchmark_output(output)

    assert results[0]["unit"] == "TFLOPS"
    assert results[0]["configs"] == [
        {"M": "512.0", "N": 512.0, "K": 512.0, "CuTile": 95.3, "PyTorch": 88.1},
        {"M": "4096.0", "N": 4096.0, "K": 4096.0, "CuTile": 612.4, "PyTorch": 590.8},
    ]


def test_two_digit_row_index():
    rows = "\n".join(f"{i:<2}  {float(2**i):6.1f}  {100.0 + i:13.1f}" for i in range(12))
    output = "dropout-performance-GBps:\n         N  CuTile (GB/s)\n" + rows + "\n"

    configs = parse_benchmark_output(output)[0]["configs"]

    assert len(configs) == 12
    assert configs[-1] == {"N": "2048.0", "CuTile": 111.0}


def test_skips_logs_and_parses_each_table():
    output = (
        "Compiling kernels...\n"
        + SOFTMAX_TRITON_3_6
        + "namespace(TILE_SIZE_M=128, TILE_SIZE_N=64): 17.5±0.2 us\n"
        + "3 succeeded, 0 failed\n"
        + "\n"
        + """\
layernorm-performance-float16-GBps:
        N  CuTile (GB/s)
0  4096.0          880.0
"""
    )
    results = parse_benchmark_output(output)

    assert [r["name"] for r in results] == ["softmax-performance-float32", "layernorm-performance-float16"]
    assert results[0]["configs"] == SOFTMAX_CONFIGS
    assert results[1]["configs"] == [{"N": "4096.0", "CuTile": 880.0}]


def test_parses_hand_aligned_table():
    # Columns that don't line up still parse when no label contains a space
    output = """\
example-TFLOPS:
     N_CTX      CuTile
0   1024.0  100.0
"""
    results = parse_benchmark_output(output)

    assert results[0]["configs"] == [{"N_CTX": "1024.0", "CuTile": 100.0}]


def test_parse_skip_reason():
    assert parse_skip_reason("Loading...\nSKIPPED: needs FP4 support\n") == "needs FP4 support"
    assert parse_skip_reason(SOFTMAX_TRITON_3_6) == ""


def test_find_benchmark_files_skips_bench_utils(monkeypatch):
    monkeypatch.delenv("BENCH_NUM_SHARDS", raising=False)

    names = {p.name for p in find_benchmark_files()}

    assert "bench_softmax.py" in names
    assert "bench_utils.py" not in names
