# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: RMS LayerNorm forward + backward (standalone)."""

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import rms_layernorm

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def reference(X, W, eps=1e-6, gemma=False):
    """PyTorch reference for RMS LayerNorm."""
    x_f32 = X.float()
    shape = x_f32.shape
    dim = shape[-1]
    x_2d = x_f32.view(-1, dim)
    var = (x_2d * x_2d).mean(dim=-1, keepdim=True)
    inv_var = torch.rsqrt(var + eps)
    normed = x_2d * inv_var
    if gemma:
        output = normed * (W.float() + 1.0)
    else:
        normed = normed.to(W.dtype)
        output = normed * W
    return output.view(*shape).to(X.dtype)


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
# Forward: sweep M (rows) with fixed dim=4096, gemma=False
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("rms-layernorm-fwd-dim4096-bf16-GBps", "M", [512, 1024, 2048, 4096], dim=4096),
    ]
)
def bench_rms_ln_fwd(M, dim, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    X = torch.randn(M, dim, dtype=dtype, device=device)
    W = torch.randn(dim, dtype=dtype, device=device)

    if backend == "pytorch":
        fn = lambda: reference(X, W, eps=1e-6)
    else:
        tilegym.set_backend(backend)
        fn = lambda: rms_layernorm(X, W, eps=1e-6, gemma=False)

    # Correctness check
    ref = lambda: reference(X, W, eps=1e-6)
    torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read X + read W + write output
    bytes_per_element = X.element_size()
    total_bytes = (M * dim + dim + M * dim) * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Backward: sweep M (rows) with fixed dim=4096, gemma=False
# Large M only — CuTile backward improves with scale
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("rms-layernorm-bwd-dim4096-bf16-GBps", "M", [4096, 8192, 16384], dim=4096),
    ]
)
def bench_rms_ln_bwd(M, dim, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    dY = torch.randn(M, dim, dtype=dtype, device=device)

    if backend == "pytorch":
        X = torch.randn(M, dim, dtype=dtype, device=device).requires_grad_(True)
        W = torch.randn(dim, dtype=dtype, device=device).requires_grad_(True)
        Y = reference(X, W, eps=1e-6)

        def fn():
            X.grad = None
            W.grad = None
            Y.backward(dY, retain_graph=True)
    else:
        tilegym.set_backend(backend)
        X = torch.randn(M, dim, dtype=dtype, device=device).requires_grad_(True)
        W = torch.randn(dim, dtype=dtype, device=device).requires_grad_(True)
        Y = rms_layernorm(X, W, eps=1e-6, gemma=False)

        def fn():
            X.grad = None
            W.grad = None
            Y.backward(dY, retain_graph=True)

    ms = triton.testing.do_bench(fn)

    # Memory: read dY + read X + read W + write dX + write dW
    bytes_per_element = dY.element_size()
    total_bytes = (2 * M * dim + dim + M * dim + dim) * bytes_per_element
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_rms_ln_fwd.run(print_data=True)
    bench_rms_ln_bwd.run(print_data=True)
