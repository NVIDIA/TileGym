# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark fused linear cross-entropy with Triton perf_report style.

Expected behavior on large BT:
- CuTile path may be slower than PyTorch in pure latency.
- CuTile path should use much less peak memory because it chunks over BT and
  avoids materializing full [BT, V] logits.
"""

import torch
import torch.nn.functional as F
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@register_impl("fused_linear_cross_entropy", "torch")
def _torch_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
    reduction: str = "mean",
    **_kwargs,
):
    del chunk_size
    logits = F.linear(hidden_states, weight, bias)
    if hidden_states.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        target = target.reshape(-1)
    return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)


ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def _supported_backends():
    return [b for b in ALL_BACKENDS if b is not None]


def _create_latency_config(hidden_size, vocab_size):
    available = _supported_backends()
    if not available:
        return None
    backends, names, styles = zip(*available)
    return triton.testing.Benchmark(
        x_names=["BT"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="Latency (ms)",
        plot_name=f"fused-lce-latency-H{hidden_size}-V{vocab_size}",
        args={
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        },
    )


def _create_memory_config(hidden_size, vocab_size):
    available = _supported_backends()
    if not available:
        return None
    backends, names, styles = zip(*available)
    return triton.testing.Benchmark(
        x_names=["BT"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="Peak Memory (MB)",
        plot_name=f"fused-lce-peakmem-H{hidden_size}-V{vocab_size}",
        args={
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        },
    )


@triton.testing.perf_report(
    [
        _create_latency_config(hidden_size=1024, vocab_size=32768),
    ]
)
def bench_fused_linear_cross_entropy_latency(BT, backend, hidden_size, vocab_size, device=DEVICE):
    dtype = torch.bfloat16

    x = torch.randn(BT, hidden_size, device=device, dtype=dtype)
    w = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    t = torch.randint(0, vocab_size, (BT,), device=device)

    def fn():
        xt = x.detach().requires_grad_(True)
        wt = w.detach().clone().requires_grad_(True)
        loss = tilegym.ops.fused_linear_cross_entropy(
            xt,
            wt,
            t,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
            backend=backend,
        )
        loss.backward()

    ms = triton.testing.do_bench(fn)
    return ms


@triton.testing.perf_report(
    [
        _create_memory_config(hidden_size=1024, vocab_size=32768),
    ]
)
def bench_fused_linear_cross_entropy_peak_memory(BT, backend, hidden_size, vocab_size, device=DEVICE):
    dtype = torch.bfloat16

    x = torch.randn(BT, hidden_size, device=device, dtype=dtype)
    w = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    t = torch.randint(0, vocab_size, (BT,), device=device)

    def run_once():
        xt = x.detach().requires_grad_(True)
        wt = w.detach().clone().requires_grad_(True)
        loss = tilegym.ops.fused_linear_cross_entropy(
            xt,
            wt,
            t,
            ignore_index=-100,
            chunk_size=512,
            reduction="mean",
            backend=backend,
        )
        loss.backward()

    for _ in range(2):
        run_once()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_once()
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() / (1024**2)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required")
    else:
        print("Note: this kernel can be slower than PyTorch but typically saves significant peak memory at large BT.")
        tilegym.set_backend("cutile" if is_backend_available("cutile") else "torch")
        bench_fused_linear_cross_entropy_latency.run(print_data=True)
        bench_fused_linear_cross_entropy_peak_memory.run(print_data=True)
