# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark: Cross Entropy Loss forward + backward (standalone)."""

import torch
import torch.nn.functional as F
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.suites.unsloth.ops import cross_entropy_loss

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def reference_fwd(logits, labels):
    """PyTorch reference for cross-entropy loss (per-sample, no reduction)."""
    return F.cross_entropy(logits, labels, reduction="none")


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
# Forward: sweep V (vocab size) with fixed (B, S)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("ce-loss-fwd-B2-S8-bf16-GBps", "V", [100000, 128256], B=2, S=8),
    ]
)
def bench_ce_loss_fwd(B, S, V, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    logits = torch.randn(B, S, V, dtype=dtype, device=device)
    labels = torch.randint(0, V, (B, S), device=device)

    if backend == "pytorch":
        logits_2d = logits.view(-1, V)
        labels_1d = labels.view(-1)
        fn = lambda: reference_fwd(logits_2d.float(), labels_1d)
    else:
        tilegym.set_backend(backend)
        fn = lambda: cross_entropy_loss(logits, labels)

    # Correctness check (compare mean; op returns reduced loss, ref returns per-sample)
    ref_logits_2d = logits.view(-1, V)
    ref_labels_1d = labels.view(-1)
    ref_out = reference_fwd(ref_logits_2d.float(), ref_labels_1d).mean()
    fn_out = fn()
    torch.testing.assert_close(fn_out.float().mean(), ref_out.float(), atol=1e-1, rtol=1e-1)

    ms = triton.testing.do_bench_cudagraph(fn)

    # Memory: read logits (B*S*V) + read labels (B*S, int64) + write loss (B*S)
    total_bytes = B * S * V * logits.element_size() + B * S * 8 + B * S * 4
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


# ---------------------------------------------------------------------------
# Backward: sweep V with fixed (B, S)
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    [
        _cfg("ce-loss-bwd-B2-S8-bf16-GBps", "V", [32000, 100000, 128256], B=2, S=8),
    ]
)
def bench_ce_loss_bwd(B, S, V, backend, device=DEVICE):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    n_rows = B * S

    if backend == "pytorch":
        logits = torch.randn(n_rows, V, dtype=dtype, device=device, requires_grad=True)
        labels = torch.randint(0, V, (n_rows,), device=device)
        loss = reference_fwd(logits.float(), labels).sum()

        def fn():
            logits.grad = None
            loss.backward(retain_graph=True)
    else:
        tilegym.set_backend(backend)
        logits = torch.randn(B, S, V, dtype=dtype, device=device, requires_grad=True)
        labels = torch.randint(0, V, (B, S), device=device)
        loss = cross_entropy_loss(logits, labels).sum()

        def fn():
            logits.grad = None
            loss.backward(retain_graph=True)

    ms = triton.testing.do_bench(fn)

    # Memory: read d_loss (B*S) + read logits (B*S*V) + write d_logits (B*S*V)
    total_bytes = 2 * n_rows * V * logits.element_size() + n_rows * 4
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_ce_loss_fwd.run(print_data=True)
    bench_ce_loss_bwd.run(print_data=True)
