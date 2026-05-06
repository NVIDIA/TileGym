# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import math
import tempfile

import torch
import triton

import tilegym
import tilegym.ops
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()
FP4_MAX = 6.0
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _is_tileir_v13_3_available() -> bool:
    """Return True if the tileiras compiler supports V_13_3 bytecode (required for fp4)."""
    try:
        from cuda.tile._bytecode.version import BytecodeVersion
        from cuda.tile._cext import dev_features_enabled
        from cuda.tile._compile import _get_max_supported_bytecode_version

        return (
            dev_features_enabled()
            and _get_max_supported_bytecode_version(tempfile.gettempdir(), allow_dev=True) >= BytecodeVersion.V_13_3
        )
    except Exception:
        return False


# No "torch" backend: there is no PyTorch equivalent for swizzled NVFP4
# quantization. The reference backend serves as the correctness baseline instead.
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
]


def get_supported_backends():
    if not _is_tileir_v13_3_available():
        return []
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["rows", "cols"],
        x_vals=[[2048, c] for c in range(2048, 50_001, 1024)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"nvfp4-quantize-{dtype_name}-GBps",
        args={"dtype": dtype},
    )


@triton.testing.perf_report([create_benchmark_config(dtype) for dtype in [torch.float32, torch.bfloat16]])
def bench_nvfp4_quantize(rows, cols, backend, dtype, device=DEVICE):
    x = torch.randn(rows, cols, dtype=dtype, device=device)
    s_enc = (FP4_MAX * FP8_E4M3_MAX) / (x.abs().max().item() + 1e-12)

    fn = lambda: tilegym.ops.nvfp4_quantize(x, s_enc=s_enc, backend=backend)

    ms = triton.testing.do_bench(fn)

    # Input read + packed output write (cols//2 bytes) + scales write (num_tiles * 512 bytes)
    num_tiles = math.ceil(rows / 128) * math.ceil(cols / 64)
    input_bytes = x.numel() * x.element_size()
    packed_bytes = rows * (cols // 2)  # uint8
    scales_bytes = num_tiles * 512  # uint8
    total_bytes = input_bytes + packed_bytes + scales_bytes

    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    if get_supported_backends():
        bench_nvfp4_quantize.run(print_data=True)
