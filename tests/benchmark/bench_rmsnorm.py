# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from bench_utils import profile_with_l2flush

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_rms_norm(
    input: torch.Tensor,
    normalized_shape: Any,
    weight: torch.Tensor,
    eps: float,
    bias: Optional[torch.Tensor] = None,
    mode: Optional[str] = None,
    **kwargs: Any,
):
    """Fused PyTorch RMSNorm baseline using F.rms_norm.

    This is the correct comparison target since it dispatches to a single
    fused kernel (the equivalent of what users would actually replace with a
    custom kernel), rather than the unfused .pow(2).mean().rsqrt() sequence
    that would artificially inflate the performance gap.
    """
    if bias is not None:
        raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")
    return F.rms_norm(input, normalized_shape, weight=weight, eps=eps)


register_impl("rms_norm", "torch")(reference_rms_norm)


# Available configs with their display names and plot styles
# (backend, mode, display label, plot style)
ALL_CONFIGS = [
    ("cutile", "static_persistent", "CuTile static persistent", ("purple", "-")),
    ("cutile", "multi_wave_reload", "CuTile multi wave reload", ("blue", "-")),
    ("cutile", "multi_wave_cached", "CuTile multi wave cached", ("red", "--")),
    ("torch", None, "PyTorch", ("green", "-")),
]


def get_supported_configs():
    cutile_available = is_backend_available("cutile")
    if cutile_available:
        return ALL_CONFIGS
    return [c for c in ALL_CONFIGS if c[0] == "torch"]


M_DEFAULT = 4096


def create_benchmark_config(dtype):
    """Create a benchmark configuration for given parameters"""
    supported = get_supported_configs()
    if not supported:
        return None

    # Use index as line_val key to pass both backend and mode
    labels = [c[2] for c in supported]
    styles = [c[3] for c in supported]
    dtype_name = str(dtype).split(".")[-1]  # e.g., 'float16' from 'torch.float16'

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # Hidden size from 1024 to 16384
        line_arg="config_idx",
        line_vals=list(range(len(supported))),
        line_names=labels,
        styles=styles,
        ylabel="GB/s",
        plot_name=f"rmsnorm-{dtype_name}-M{M_DEFAULT}",
        args={
            "dtype": dtype,
            "M": M_DEFAULT,
        },  # Fixed batch*seq_len
    )


@triton.testing.perf_report([create_benchmark_config(dtype) for dtype in [torch.float16, torch.bfloat16]])
def bench_rmsnorm(N, config_idx, dtype, M, device=DEVICE):
    eps = 1e-5

    # Create input tensors
    x_shape = (M, N)
    w_shape = (N,)

    x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
    weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

    supported = get_supported_configs()
    backend, mode, _, _ = supported[config_idx]

    fn = lambda: tilegym.ops.rms_norm(x, w_shape, weight, eps, mode=mode, backend=backend)
    ref = lambda: reference_rms_norm(x, w_shape, weight, eps)
    torch.testing.assert_close(fn(), ref(), atol=5e-2, rtol=0.0)

    # Benchmark the function
    ms = profile_with_l2flush(fn)

    # Calculate memory bandwidth (GB/s)
    # RMSNorm operation: read input, read weight, write output
    # Memory access: read x + read weight + write output
    bytes_per_element = x.element_size()

    input_bytes = x.numel() * bytes_per_element  # Read input
    weight_bytes = weight.numel() * bytes_per_element  # Read weight
    output_bytes = x.numel() * bytes_per_element  # Write output

    total_bytes = input_bytes + weight_bytes + output_bytes

    # Convert to GB/s
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)

    return gb_per_s


if __name__ == "__main__":
    bench_rmsnorm.run(print_data=True)
