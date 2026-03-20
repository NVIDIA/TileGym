# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark forward-only fused linear cross-entropy with Triton perf_report style."""

import torch
import torch.nn.functional as F
import triton

from tilegym.backend import is_backend_available

if is_backend_available("cutile"):
    from tilegym.ops.cutile import fused_linear_cross_entropy
else:
    fused_linear_cross_entropy = None

DEVICE = triton.runtime.driver.active.get_active_torch_device()

ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def _supported_backends():
    return [b for b in ALL_BACKENDS if b is not None]


def _torch_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    logits = F.linear(hidden_states, weight, bias)
    if hidden_states.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        target = target.reshape(-1)
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


def _create_tflops_config(hidden_size, vocab_size, datatype):
    available = _supported_backends()
    if not available:
        return None
    backends, names, styles = zip(*available)
    dtype_name = str(datatype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["BT"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"fused-lce-forward-H{hidden_size}-V{vocab_size}-{dtype_name}-TFLOPS",
        args={
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "datatype": datatype,
        },
    )


def _create_memory_gbps_config(hidden_size, vocab_size, datatype):
    available = _supported_backends()
    if not available:
        return None
    backends, names, styles = zip(*available)
    dtype_name = str(datatype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["BT"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"fused-lce-memory-H{hidden_size}-V{vocab_size}-{dtype_name}-GBps",
        args={
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        _create_tflops_config(hidden_size=1024, vocab_size=32768, datatype=datatype)
        for datatype in [torch.bfloat16, torch.float16]
    ]
)
def bench_fused_linear_cross_entropy(BT, backend, hidden_size, vocab_size, datatype, device=DEVICE):
    x = torch.randn(BT, hidden_size, device=device, dtype=datatype)
    w = torch.randn(vocab_size, hidden_size, device=device, dtype=datatype)
    t = torch.randint(0, vocab_size, (BT,), device=device)

    if backend == "cutile":
        fn = lambda: fused_linear_cross_entropy(
            x,
            w,
            t,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
        )
    else:
        fn = lambda: _torch_fused_linear_cross_entropy(x, w, t, ignore_index=-100, reduction="mean")

    ms = triton.testing.do_bench(fn)
    flops = 2 * BT * hidden_size * vocab_size
    return flops * 1e-12 / (ms * 1e-3)


@triton.testing.perf_report(
    [
        _create_memory_gbps_config(hidden_size=1024, vocab_size=32768, datatype=datatype)
        for datatype in [torch.bfloat16, torch.float16]
    ]
)
def bench_fused_linear_cross_entropy_memory(BT, backend, hidden_size, vocab_size, datatype, device=DEVICE):
    x = torch.randn(BT, hidden_size, device=device, dtype=datatype)
    w = torch.randn(vocab_size, hidden_size, device=device, dtype=datatype)
    t = torch.randint(0, vocab_size, (BT,), device=device)

    if backend == "cutile":
        fn = lambda: fused_linear_cross_entropy(
            x,
            w,
            t,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
        )
    else:
        fn = lambda: _torch_fused_linear_cross_entropy(x, w, t, ignore_index=-100, reduction="mean")

    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    ms = triton.testing.do_bench(fn)
    return peak_memory_bytes * 1e-9 / (ms * 1e-3)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
    else:
        bench_fused_linear_cross_entropy.run(print_data=True)
        bench_fused_linear_cross_entropy_memory.run(print_data=True)
