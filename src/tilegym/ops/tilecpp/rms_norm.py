# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ RMS Normalization Operation
Root mean square normalization using CUDA C++ tile kernels compiled with nvcc.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_dtype_info
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# Define kernels
_rms_norm_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "rms_norm.cuh",
    kernel_name="rms_norm_kernel",
)

_rms_norm_kernel_pv = TileCppKernel(
    source_path=Path(__file__).parent / "rms_norm.cuh",
    kernel_name="rms_norm_kernel_pv",
)

_rms_norm_static_persistent_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "rms_norm.cuh",
    kernel_name="rms_norm_static_persistent_kernel",
)

_rms_norm_backward_dx_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "rms_norm.cuh",
    kernel_name="rms_norm_backward_dx_kernel",
)

# Supported block sizes (must be power of 2)
_BLOCK_SIZES = [256, 512, 1024, 2048, 4096]


def _get_block_size(N: int) -> int:
    """Get the smallest block size that can handle N columns."""
    for bs in _BLOCK_SIZES:
        if bs >= N:
            return bs
    # Fall back to largest
    return _BLOCK_SIZES[-1]


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _get_num_sm():
    """Get number of SMs on the current GPU."""
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _ceildiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


def _launch_rms_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    stride: int,
    N: int,
    eps: float,
    M: int,
    block_size: int,
):
    dump_kernel_types("rms_norm_kernel", x, weight, y, rstd)
    dtype = x.dtype

    eps_tp = f"{eps}f"

    # Signature: const T*, const T*, T*, float*, int
    kernel, _, _ = _rms_norm_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size, N, eps_tp],
        signature="const {T}*, const {T}*, {T}*, float*, int",
    )

    grid = M

    _rms_norm_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(y.data_ptr()),
            np.uint64(rstd.data_ptr()),
            np.int32(stride),
        ],
    )


def _launch_rms_norm_kernel_pv(
    x: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    M: int,
    N: int,
    eps: float,
    block_size: int,
):
    """Launch the rms_norm_kernel_pv CUDA kernel (unmasked partition_view version).

    Uses tensor_span + partition_view with unmasked loads for N == BLOCK_SIZE.
    """
    dump_kernel_types("rms_norm_kernel_pv", x, weight, y, rstd)
    dtype = x.dtype

    # Template params: M, N, BLOCK_SIZE
    kernel, _, _ = _rms_norm_kernel_pv.get_kernel(
        dtype=dtype,
        template_params=[M, N, block_size],
        signature="const {T}*, const {T}*, {T}*, float*, float",
    )

    # One block per row
    grid = M

    _rms_norm_kernel_pv.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(y.data_ptr()),
            np.uint64(rstd.data_ptr()) if rstd is not None else np.uint64(0),
            np.float32(eps),
        ],
    )


def _launch_rms_norm_static_persistent_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    M: int,
    N: int,
    eps: float,
    offset: float,
    tile_size_m: int,
    tile_size_n: int,
    occupancy: int = 1,
):
    """
    Template params: T, TILE_SIZE_M, TILE_SIZE_N, occupancy, M, N, NUM_SMS, EPS, OFFSET.
    Runtime args  : X, Y, W, Rstd.
    Grid          : min(NUM_SMS, ceil(M/TILE_M) * ceil(N/TILE_N)).
    """
    dump_kernel_types("rms_norm_static_persistent_kernel", x, y, weight, rstd)
    dtype = x.dtype

    NUM_SMS = _get_num_sm()
    grid_size = min(NUM_SMS, _ceildiv(M, tile_size_m) * _ceildiv(N, tile_size_n))

    # nvcc-style float-as-template-param: encode as the literal text the
    # template instantiation expects (e.g. ``9.99999974e-06f``).
    eps_tp = f"{eps}f"
    offset_tp = f"{offset}f"

    kernel, _, _ = _rms_norm_static_persistent_kernel.get_kernel(
        dtype=dtype,
        template_params=[tile_size_m, tile_size_n, occupancy, M, N, NUM_SMS, eps_tp, offset_tp],
        signature="const {T}*, {T}*, const {T}*, float*",
    )

    _rms_norm_static_persistent_kernel.launch(
        grid=grid_size,
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(y.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(rstd.data_ptr()),
        ],
    )


def _launch_rms_norm_backward_dx_kernel(
    dx: torch.Tensor,
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    temp_buffer: torch.Tensor,
    stride: int,
    N: int,
    M: int,
    block_size: int,
):
    """Launch the rms_norm_backward_dx_kernel CUDA kernel."""
    dump_kernel_types("rms_norm_backward_dx_kernel", dx, dy, x, weight)
    dtype = x.dtype

    # Signature: T*, const T*, const T*, const T*, const float*, float*, int, int
    kernel, _, _ = _rms_norm_backward_dx_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="{T}*, const {T}*, const {T}*, const {T}*, const float*, float*, int, int",
    )

    # One block per row
    grid = M

    _rms_norm_backward_dx_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(dx.data_ptr()),
            np.uint64(dy.data_ptr()),
            np.uint64(x.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(rstd.data_ptr()),
            np.uint64(temp_buffer.data_ptr()),
            np.int32(stride),
            np.int32(N),
        ],
    )


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone RMSNorm backward pass using CUDA Tile C++ kernels.

    Computes dx and dw given the upstream gradient dy, the original input x,
    the weight, and the saved rstd from the forward pass.

    Args:
        x: Original input tensor
        dy: Upstream gradient tensor
        weight: Weight tensor of shape (N,)
        rstd: Reciprocal standard deviation per row (M,)

    Returns:
        Tuple of (dx, dw) gradients
    """
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    rstd = rstd.contiguous()

    x_shape = x.shape

    # Flatten to [M, N]
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])

    M, N = x.shape

    # Allocate outputs
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)  # shape (N,)
    temp_buffer = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    dx = dx.detach()
    dw = dw.detach()

    block_size = _next_power_of_2(N)
    if block_size > 4096:
        block_size = 4096

    # Launch dx kernel (also populates temp_buffer with dy*x*rstd for dw computation)
    _launch_rms_norm_backward_dx_kernel(
        dx,
        dy,
        x,
        weight,
        rstd,
        temp_buffer,
        x.stride(0),
        N,
        M,
        block_size,
    )

    dw = temp_buffer[:, :N].to(torch.float32).sum(dim=0).to(weight.dtype)

    # Reshape dx back, dw already correct
    return dx.view(*x_shape), dw


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        eps,
        bias=None,
        mode=None,
    ):
        """
        CUDA Tile C++ RMSNorm forward pass.

        Args:
            x: Input tensor of shape [M, N]
            normalized_shape: Normalization shape (for compatibility, not used)
            weight: Weight tensor of shape [N]
            eps: Epsilon value for numerical stability
            bias: Bias tensor of shape [N], default is None (not supported)
            mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload").
                  ``multi_wave_cached`` is not implemented in the tilecpp backend
                  and raises NotImplementedError.

        Returns:
            Normalized and transformed tensor of same shape as input
        """
        if bias is not None:
            raise NotImplementedError("Bias is not supported in TileCpp RMSNorm")

        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()

        # Reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])

        # Allocate output tensor
        y = torch.empty_like(x)
        M, N = x_arg.shape

        NUM_SMS = _get_num_sm()
        if mode is None:
            mode = "static_persistent" if M > NUM_SMS * 2 else "multi_wave_reload"

        if mode == "static_persistent":
            # TILE_SIZE_N = next power of 2 >= N (kernel loads the full row at once).
            TILE_SIZE_N = _next_power_of_2(N)

            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2
            else:
                TILE_SIZE_M = 4

            # rstd is saved for the backward pass.
            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

            _launch_rms_norm_static_persistent_kernel(
                x_arg,
                y.reshape(-1, y.shape[-1]),
                weight,
                rstd,
                M,
                N,
                eps,
                0.0,  # offset: 0.0 for standard RMSNorm
                TILE_SIZE_M,
                TILE_SIZE_N,
            )

            ctx.save_for_backward(x, weight, rstd)
            ctx.eps = eps
            ctx.block_size = TILE_SIZE_N
        elif mode == "multi_wave_reload":
            MAX_FUSED_SIZE = 4096 // x_arg.element_size()
            block_size = min(MAX_FUSED_SIZE, _next_power_of_2(N))

            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

            _launch_rms_norm_kernel(
                x_arg,
                weight,
                y.reshape(-1, y.shape[-1]),
                rstd,
                x_arg.stride(0),
                N,
                eps,
                M,
                block_size,
            )

            ctx.save_for_backward(x, weight, rstd)
            ctx.eps = eps
            ctx.block_size = block_size
        elif mode == "multi_wave_cached":
            raise NotImplementedError("multi_wave_cached mode is not implemented for the tilecpp backend")
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: None, 'static_persistent', "
                f"'multi_wave_reload', 'multi_wave_cached'"
            )

        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for CUDA Tile C++ RMSNorm.
        Retrieves saved tensors and delegates to rms_norm_backward().
        """
        x, weight, rstd = ctx.saved_tensors

        # Call the standalone backward function
        dx, dw = rms_norm_backward(x, dy, weight, rstd)

        # Return gradients: (x, normalized_shape, weight, eps, bias, mode)
        return dx, None, dw, None, None, None


@register_impl("rms_norm", backend="tilecpp")
def rms_norm(input, normalized_shape, weight, eps, bias=None, mode=None, **kwargs):
    """
    Root mean square normalization implemented using CUDA Tile C++

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation prior to division
        bias: Bias tensor of shape (N,), default is None (not supported)
        mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload").
              ``multi_wave_cached`` is not implemented in the tilecpp backend
              and raises NotImplementedError.
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    return RMSNorm.apply(input, normalized_shape, weight, eps, bias, mode)


class TileCppRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm implementation using CUDA Tile C++ kernels for faster computation

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon value for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states, mode=None):
        """
        Forward pass with optional mode override

        Args:
            hidden_states: Input tensor
            mode: Default is None, which means use heuristic to
                  decide which kernel mode to use for better performance
        """
        return rms_norm(
            hidden_states,
            None,
            self.weight,
            self.variance_epsilon,
            mode=mode,
        )

    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    @staticmethod
    def compute_rstd_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute rstd (reciprocal standard deviation) for RMSNorm using PyTorch.
        Simulates what the forward pass would save for backward."""
        x_2d = x.reshape(-1, x.shape[-1])
        x_fp32 = x_2d.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1)
        rstd = torch.rsqrt(variance + eps)
        return rstd

    @staticmethod
    def rms_norm_backward(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Only for testing purposes.
        Calls the CUDA Tile C++ backward implementation.
        """
        return rms_norm_backward(x, dy, weight, rstd)

    @staticmethod
    def rms_norm_backward_torch(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standalone RMSNorm backward pass using PyTorch.
        This is explicitly the torch reference implementation, not the tilecpp implementation."""
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1])
        M, N = x.shape

        # Reshape rstd for broadcasting: (M,) -> (M, 1)
        rstd = rstd.view(M, 1)

        # Gradient w.r.t. weight: sum over batch dimension (accumulate in float32).
        dw = ((x.float() * dy.float()) * rstd).sum(dim=0, dtype=torch.float32)

        # Normalized x (before scaling by weight) - for dx computation
        x_norm = x * rstd

        # Gradient w.r.t. x (accumulate in float32)
        dy_weighted = dy * weight
        c1 = (dy_weighted * x_norm).sum(
            dim=1, keepdim=True, dtype=torch.float32
        )  # ensure accumulates are done in float32 to avoid precision issues
        dx = rstd * (dy_weighted - x_norm * c1 / N)

        dx = dx.view(x_shape).to(x.dtype)
        dw = dw.to(weight.dtype)

        return dx, dw

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@register_impl("get_rms_norm_module", backend="tilecpp")
def get_rms_norm_module(model: str = "llama"):
    if model != "llama":
        raise NotImplementedError(
            f"tilecpp RMSNorm currently only supports model='llama' (got model={model!r}). "
            "Gemma3-style RMSNorm with non-zero offset is not implemented."
        )
    return TileCppRMSNorm
