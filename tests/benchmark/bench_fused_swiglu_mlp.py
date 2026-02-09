#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
End-to-end benchmark for PartiallyFusedSwiGLUMLP vs PyTorch LlamaMLP.
Tests full MLP forward/backward including matmul overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available
from tilegym.ops.fused_swiglu import PartiallyFusedSwiGLUMLP


class MockConfig:
    """Mock config for MLP modules."""

    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"


class PyTorchSwiGLUMLP(nn.Module):
    """Reference PyTorch SwiGLU MLP (same structure as LlamaMLP)."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# Model configurations
MODEL_CONFIGS = {
    "llama-7b": {"hidden_size": 4096, "intermediate_size": 11008},
    "llama-13b": {"hidden_size": 5120, "intermediate_size": 13824},
    "llama-70b": {"hidden_size": 8192, "intermediate_size": 28672},
}


def get_providers():
    providers = [("torch", "PyTorch", ("green", "-"))]
    if is_backend_available("cutile"):
        providers.insert(0, ("cutile", "CuTile", ("orange", "-")))
    return providers


def create_benchmark_config(mode, model_name, batch_size):
    providers = get_providers()
    if not providers:
        return None

    backends, names, styles = zip(*providers)
    mode_name = mode.replace("_", "-")

    return triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128, 256, 512, 1024, 2048],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"fused-swiglu-mlp-{mode_name}-{model_name}-bs{batch_size}-GBps",
        args={
            "model_name": model_name,
            "batch_size": batch_size,
            "mode": mode,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(mode, model_name, batch_size)
        for mode in ["forward", "backward", "full"]
        for model_name in ["llama-7b"]  # Start with 7B to avoid OOM
        for batch_size in [1, 4]
    ]
)
def bench_fused_swiglu_mlp(
    seq_len,
    model_name,
    batch_size,
    backend,
    mode,
    device="cuda",
    dtype=torch.bfloat16,
):
    config_params = MODEL_CONFIGS[model_name]
    config = MockConfig(**config_params)
    H = config.hidden_size
    I = config.intermediate_size

    # Create MLP module
    if backend == "cutile":
        tilegym.set_backend("cutile")
        mlp = PartiallyFusedSwiGLUMLP(config).to(device).to(dtype)
    else:
        mlp = PyTorchSwiGLUMLP(config).to(device).to(dtype)

    # Create input
    x = torch.randn(batch_size, seq_len, H, dtype=dtype, device=device, requires_grad=True)
    bytes_per_element = x.element_size()
    M = batch_size * seq_len

    def fwd():
        return mlp(x)

    # Memory calculation for SwiGLU MLP:
    # Forward: x(M,H) -> gate(M,I), up(M,I) -> glu(M,I) -> out(M,H)
    # Backward: similar pattern in reverse
    fwd_bytes = M * (H + 2 * I + I + H) * bytes_per_element
    bwd_bytes = fwd_bytes * 2  # Approximate: gradients have similar memory footprint

    if mode == "forward":
        total_bytes = fwd_bytes
        ms = triton.testing.do_bench(fwd, rep=10)
    elif mode == "backward":
        y = fwd()
        dy = torch.randn_like(y)
        total_bytes = bwd_bytes
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), rep=10)
    else:  # full
        dy = torch.randn(batch_size, seq_len, H, dtype=dtype, device=device)
        total_bytes = fwd_bytes + bwd_bytes

        def full():
            y = fwd()
            y.backward(dy, retain_graph=True)

        ms = triton.testing.do_bench(full, rep=10)

    # Calculate GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_fused_swiglu_mlp.run(print_data=True)
