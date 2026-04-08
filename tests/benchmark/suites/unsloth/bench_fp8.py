# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: FP8 W8A8 block matmul + weight dequant (standalone)."""

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import w8a8_block_fp8_matmul
from tilegym.suites.unsloth.ops import weight_dequant

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# FP8 kernels have no PyTorch reference -- cutile backend only.
ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
]


def _supported():
    return [b for b in ALL_BACKENDS if b is not None]


def _cfg(plot_name, x_name, x_vals, **extra):
    avail = _supported()
    if not avail:
        return None
    backends, names, styles = zip(*avail)
    return triton.testing.Benchmark(
        x_names=[x_name],
        x_vals=x_vals,
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=plot_name,
        args=extra,
    )


# ---------------------------------------------------------------------------
# W8A8 block FP8 matmul: sweep K with fixed M, N, block_size=[128,128]
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg(
            "fp8-w8a8-matmul-M1024-N2048-blk128-GBps",
            "K",
            [256, 512, 1024, 2048],
            M=1024,
            N=2048,
            block_size_n=128,
            block_size_k=128,
        ),
    ]
)
def bench_w8a8_matmul(M, N, K, block_size_n, block_size_k, backend, device=DEVICE):
    torch.manual_seed(42)
    block_size = [block_size_n, block_size_k]

    A_f32 = torch.randn(M, K, device=device) * 0.1
    A = A_f32.to(torch.float8_e4m3fn)
    B_f32 = torch.randn(N, K, device=device) * 0.1
    B = B_f32.to(torch.float8_e4m3fn)

    As = torch.rand(M, K // block_size_k, dtype=torch.float32, device=device) + 0.1
    Bs = torch.rand(N // block_size_n, K // block_size_k, dtype=torch.float32, device=device) + 0.1

    tilegym.set_backend(backend)
    fn = lambda: w8a8_block_fp8_matmul(A, B, As, Bs, block_size=block_size)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read A (fp8) + read B (fp8) + read As + read Bs + write C (bf16)
    total_bytes = (M * K + N * K) * 1 + As.numel() * 4 + Bs.numel() * 4 + M * N * 2
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Weight dequantization: sweep N with fixed M=2048, block_size=128
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("fp8-weight-dequant-M2048-blk128-GBps", "N", [512, 1024, 2048, 4096], M=2048, block_size=128),
    ]
)
def bench_weight_dequant(M, N, block_size, backend, device=DEVICE):
    torch.manual_seed(42)
    x_f32 = torch.randn(M, N, device=device)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    bm = (M + block_size - 1) // block_size
    bn = (N + block_size - 1) // block_size
    s = torch.rand(bm, bn, dtype=torch.float32, device=device) + 0.1

    tilegym.set_backend(backend)
    fn = lambda: weight_dequant(x_fp8, s, block_size=block_size)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read x_fp8 (1 byte) + read scale + write output (bf16, 2 bytes)
    total_bytes = M * N * 1 + bm * bn * 4 + M * N * 2
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_w8a8_matmul.run(print_data=True)
    bench_weight_dequant.run(print_data=True)
