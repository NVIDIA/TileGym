# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: SwiGLU forward + backward (standalone)."""

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import swiglu_bwd
from tilegym.suites.unsloth.ops import swiglu_fg

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch references
# ---------------------------------------------------------------------------


def reference_fwd(e, g):
    e_f32 = e.float()
    f = e_f32 * torch.sigmoid(e_f32)
    f = f.to(g.dtype)
    return f * g


def reference_bwd(DW, e, g):
    e_f32 = e.float()
    se = torch.sigmoid(e_f32)
    f = se * e_f32
    f = f.to(DW.dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = dg.float() * se * (1.0 + e_f32 * (1.0 - se))
    de = de.to(DW.dtype)
    return h, df, de


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("pytorch", "PyTorch", ("green", "-")),
]


def _supported():
    return [b for b in ALL_BACKENDS if b is not None]


def _cfg(plot_name, x_vals, **extra):
    avail = _supported()
    if not avail:
        return None
    backends, names, styles = zip(*avail)
    return triton.testing.Benchmark(
        x_names=["N"],
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
# Forward: sweep N with fixed (B, M)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("swiglu-fwd-B4-M512-bf16-GBps", [2048, 4096, 5120, 8192], B=4, M=512),
    ]
)
def bench_swiglu_fwd(B, M, N, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    e = torch.randn(B, M, N, dtype=dtype, device=device)
    g = torch.randn(B, M, N, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference_fwd(e, g)
    else:
        tilegym.set_backend(backend)
        fn = lambda: swiglu_fg(e, g)

    # Correctness check
    ref = lambda: reference_fwd(e, g)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read e + read g + write output = 3 tensors
    total_bytes = 3 * B * M * N * e.element_size()
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Backward: sweep N with fixed (B, M)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("swiglu-bwd-B4-M512-bf16-GBps", [2048, 4096, 5120, 8192], B=4, M=512),
    ]
)
def bench_swiglu_bwd(B, M, N, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    DW = torch.randn(B, M, N, dtype=dtype, device=device)
    e = torch.randn(B, M, N, dtype=dtype, device=device)
    g = torch.randn(B, M, N, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference_bwd(DW.clone(), e.clone(), g.clone())
    else:
        tilegym.set_backend(backend)
        fn = lambda: swiglu_bwd(DW.clone(), e.clone(), g.clone())

    # Correctness check
    ref = lambda: reference_bwd(DW.clone(), e.clone(), g.clone())
    result = fn()
    expected = ref()
    for r, ex in zip(result, expected):
        torch.testing.assert_close(r, ex, atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read DW + read e + read g + write h + write df + write de = 6 tensors
    total_bytes = 6 * B * M * N * DW.element_size()
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_swiglu_fwd.run(print_data=True)
    bench_swiglu_bwd.run(print_data=True)
