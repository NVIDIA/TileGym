# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: GEGLU approx forward + backward (standalone)."""

import math

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import geglu_approx_backward
from tilegym.suites.unsloth.ops import geglu_approx_forward

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch references
# ---------------------------------------------------------------------------


def reference_approx_fwd(gate, up):
    g_f32 = gate.float()
    s = math.sqrt(2.0 / math.pi)
    f = 0.5 * g_f32 * (torch.tanh(s * g_f32 * (1.0 + 0.044715 * g_f32 * g_f32)) + 1.0)
    f = f.to(up.dtype)
    return f * up


def reference_approx_bwd(DW, e, g):
    e_f32 = e.float()
    s = math.sqrt(2.0 / math.pi)
    a = s * e_f32
    b = a * 0.044715 * e_f32 * e_f32
    T = 1.0 + torch.tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2
    f = T2 * e_f32
    f = f.to(DW.dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = dg.float() * df_de
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
# Approx forward: sweep N with fixed (B, M)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("geglu-approx-fwd-B4-M512-bf16-GBps", "N", [2048, 4096, 5120, 8192], B=4, M=512),
    ]
)
def bench_geglu_approx_fwd(B, M, N, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    gate = torch.randn(B, M, N, dtype=dtype, device=device)
    up = torch.randn(B, M, N, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference_approx_fwd(gate, up)
    else:
        tilegym.set_backend(backend)
        fn = lambda: geglu_approx_forward(gate, up)

    # Correctness check
    ref = lambda: reference_approx_fwd(gate, up)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read gate + read up + write output = 3 tensors
    total_bytes = 3 * B * M * N * gate.element_size()
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Approx backward: sweep N with fixed (B, M)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("geglu-approx-bwd-B4-M512-bf16-GBps", "N", [2048, 4096, 5120, 8192], B=4, M=512),
    ]
)
def bench_geglu_approx_bwd(B, M, N, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    DW = torch.randn(B, M, N, dtype=dtype, device=device)
    e = torch.randn(B, M, N, dtype=dtype, device=device)
    g = torch.randn(B, M, N, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference_approx_bwd(DW.clone(), e.clone(), g.clone())
    else:
        tilegym.set_backend(backend)
        fn = lambda: geglu_approx_backward(DW.clone(), e.clone(), g.clone())

    # Correctness check
    ref = lambda: reference_approx_bwd(DW.clone(), e.clone(), g.clone())
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
    bench_geglu_approx_fwd.run(print_data=True)
    bench_geglu_approx_bwd.run(print_data=True)
