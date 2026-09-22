# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel

from .utils import cached_replace_hints
from .utils import next_power_of_2

ConstInt = ct.Constant[int]
_LOG2_E = 1.4426950408889634

_SILU_OCCUPANCY_CONFIGS = tuple(SimpleNamespace(occupancy=occupancy) for occupancy in (None, 1, 2, 4, 8, 12, 16))
_SILU_TUNE_CACHE: dict = {}


# To be launched with grid = number of rows (batch_size)
# each "block" computes an entire row of the ouptut
@ct.kernel
def _silu_and_mul_kernel_row_wise(
    input,
    output,
    TILE_SIZE: ConstInt,
    TOTAL_HIDDEN_SIZE: ConstInt,
):
    bid = ct.bid(0)  # this gives us our row
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    # For 2D input (batch_size, 2*hidden_size), we need 2D indices
    # Row index is just bid (scalar), column indices are offsets-based
    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size)
    b_col_idx = offsets + TOTAL_HIDDEN_SIZE  # Second half: [hidden_size, 2*hidden_size)
    # Load tiles using gather with 2D indices
    # gather broadcasts (scalar, tile) to (tile,)
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Implement sigmoid for SiLU
    denom = 1 + ct.exp2(ct.mul(-a_tile, _LOG2_E, flush_to_zero=True), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # Perform SiLU(a) * b
    silu_a = a_tile * sigmoid_a
    result = silu_a * b_tile
    result = ct.astype(result, output.dtype)
    # output is also 2D: (batch_size, hidden_size)
    out_col_idx = offsets
    ct.scatter(output, (row_idx, out_col_idx), result, check_bounds=True)


# Backward kernel for silu_and_mul
# Computes gradients using recomputation (no saved activations)
# Forward: c = silu(a) * b = a * sigmoid(a) * b
# da = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
#    = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
# db = dc * silu(a)
@experimental_kernel
@ct.kernel
def _silu_and_mul_backward_kernel_row_wise(
    grad_output,  # dc: (batch_size, hidden_size)
    input,  # original input: (batch_size, 2*hidden_size)
    grad_a,  # output: (batch_size, hidden_size)
    grad_b,  # output: (batch_size, hidden_size)
    TILE_SIZE: ConstInt,
    TOTAL_HIDDEN_SIZE: ConstInt,
):
    bid = ct.bid(0)  # row index
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size)
    b_col_idx = offsets + TOTAL_HIDDEN_SIZE  # Second half: [hidden_size, 2*hidden_size)

    # Load grad_output (dc)
    dc_tile = ct.gather(grad_output, (row_idx, offsets), check_bounds=True)
    dc_tile = ct.astype(dc_tile, torch.float32)

    # Recompute a and b from input (saves memory vs saving in forward)
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Recompute sigmoid(a) and silu(a)
    denom = 1 + ct.exp2(ct.mul(-a_tile, _LOG2_E, flush_to_zero=True), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)
    silu_a = a_tile * sigmoid_a

    # Compute db = dc * silu(a)
    db_tile = dc_tile * silu_a
    db_tile = ct.astype(db_tile, input.dtype)
    ct.scatter(grad_b, (row_idx, offsets), db_tile, check_bounds=True)

    # Compute da = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    # = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
    one_minus_sigmoid = 1.0 + -sigmoid_a
    silu_grad = sigmoid_a + silu_a * one_minus_sigmoid
    da_tile = dc_tile * (b_tile * silu_grad)
    da_tile = ct.astype(da_tile, input.dtype)
    ct.scatter(grad_a, (row_idx, offsets), da_tile, check_bounds=True)


def _ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


def _silu_autotune_hints(config, launch_hints):
    hints = launch_hints.copy()
    for name in ("occupancy", "num_ctas", "num_worker_warps"):
        value = getattr(config, name, None)
        if value is not None:
            hints[name] = value
    return hints


def _silu_autotune_settings(tile_size, dtype, device):
    tile_bytes = tile_size * dtype.itemsize
    capability = torch.cuda.get_device_capability(device)
    if tile_bytes <= 2048:
        hints = {"num_ctas": 1, "num_worker_warps": 4}
    elif tile_bytes <= 8192:
        hints = {"num_ctas": 1, "num_worker_warps": 8}
    elif capability[0] < 9:
        hints = {"num_ctas": 1, "num_worker_warps": 8}
    else:
        hints = {"num_ctas": 2, "num_worker_warps": 8}
    return _SILU_OCCUPANCY_CONFIGS, hints


def _launch_silu_kernel(kernel, args, grid, cache_key, configs, launch_hints):
    stream = torch.cuda.current_stream(args[0].device)
    if is_autotune_disabled():
        tuned_kernel = kernel
    else:
        tuned_kernel = _SILU_TUNE_CACHE.get(cache_key)
        if tuned_kernel is None:
            result = exhaustive_search(
                configs,
                stream,
                lambda _: grid,
                kernel,
                lambda _: args,
                lambda config: _silu_autotune_hints(config, launch_hints),
            )
            best_hints = _silu_autotune_hints(result.best.config, launch_hints)
            tuned_kernel = kernel if not best_hints else cached_replace_hints(kernel, **best_hints)
            _SILU_TUNE_CACHE[cache_key] = tuned_kernel

    ct.launch(stream, grid, tuned_kernel, args)


def _silu_and_mul_forward(input_flat, output, hidden_size):
    tile_size = next_power_of_2(hidden_size)
    args = (input_flat, output, tile_size, hidden_size)
    cache_key = ("fwd", hidden_size, input_flat.dtype, str(input_flat.device))
    configs, launch_hints = _silu_autotune_settings(tile_size, input_flat.dtype, input_flat.device)

    if torch.cuda.get_device_capability(input_flat.device) == (10, 7):
        tile_bytes = tile_size * input_flat.dtype.itemsize
        if tile_bytes <= 8192:
            launch_hints = {"num_ctas": 1, "num_worker_warps": 4}
        elif tile_bytes <= 16384:
            launch_hints = {"num_ctas": 2, "num_worker_warps": 4}

    _launch_silu_kernel(_silu_and_mul_kernel_row_wise, args, (input_flat.shape[0],), cache_key, configs, launch_hints)


def _silu_and_mul_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for fused SiLU and Mul operation.

    Args:
        grad_output: Gradient w.r.t. output, shape (..., hidden_size)
        input: Original input tensor, shape (..., 2*hidden_size)

    Returns:
        Tuple of (grad_a, grad_b) each with shape (..., hidden_size)
    """
    original_output_shape = grad_output.shape
    hidden_size = original_output_shape[-1]

    # Flatten to 2D
    grad_output_flat = grad_output.contiguous().view(-1, hidden_size)
    input_flat = input.contiguous().view(-1, input.shape[-1])
    batch_size = grad_output_flat.shape[0]
    grad_a = torch.empty_like(grad_output_flat)
    grad_b = torch.empty_like(grad_output_flat)

    tile_size = next_power_of_2(hidden_size)
    grid = (batch_size,)
    args = (grad_output_flat, input_flat, grad_a, grad_b, tile_size, hidden_size)
    cache_key = ("bwd", hidden_size, grad_output_flat.dtype, str(grad_output_flat.device))
    configs, launch_hints = _silu_autotune_settings(tile_size, grad_output_flat.dtype, grad_output_flat.device)
    _launch_silu_kernel(_silu_and_mul_backward_kernel_row_wise, args, grid, cache_key, configs, launch_hints)

    return grad_a.view(*original_output_shape), grad_b.view(*original_output_shape)


class _SiLUAndMulFunction(torch.autograd.Function):
    """Autograd function for silu_and_mul with backward support."""

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Save input for backward (used in recomputation)
        ctx.save_for_backward(input)

        # Get shape info
        original_shape = input.shape
        hidden_size = original_shape[-1] // 2

        # Flatten input to 2D
        input_flat = input.view(-1, original_shape[-1])
        batch_size = input_flat.shape[0]
        output = torch.empty(
            (batch_size, hidden_size),
            dtype=input.dtype,
            device=input.device,
        )

        _silu_and_mul_forward(input_flat, output, hidden_size)

        # Reshape output
        output_shape = list(original_shape)
        output_shape[-1] = hidden_size
        return output.view(*output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_a, grad_b = _silu_and_mul_backward(grad_output, input)

        # Concatenate gradients for the original input layout
        grad_input = torch.cat([grad_a, grad_b], dim=-1)
        return grad_input


@register_impl("silu_and_mul", backend="cutile")
@_ensure_contiguous
def silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused SiLU and Mul operation implemented with Cutile.

    Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
        out (Optional[torch.Tensor]): Output tensor, if specified kernel will update in-place
    Returns:
        torch.Tensor: Output tensor of shape (..., hidden_size)
    """
    # Use autograd wrapper when backward is needed
    if input.requires_grad:
        if out is not None:
            raise ValueError("out parameter not supported when requires_grad=True")
        return _SiLUAndMulFunction.apply(input)

    # Direct kernel call for inference (no backward needed)
    original_shape = input.shape
    hidden_size = original_shape[-1] // 2

    # Flatten input to 2D: (batch_size, 2 * hidden_size)
    input_flat = input.view(-1, original_shape[-1])
    batch_size = input_flat.shape[0]

    # Get final output shape
    output_shape = list(original_shape)
    output_shape[-1] = hidden_size
    # Prepare output tensor
    if out is not None:
        # Ensure out shape is correct
        if out.shape != tuple(output_shape):
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {tuple(output_shape)}")
        output = out.view(-1, hidden_size)
    else:
        output = torch.empty(
            (batch_size, hidden_size),
            dtype=input.dtype,
            device=input.device,
        )

    _silu_and_mul_forward(input_flat, output, hidden_size)
    return output.reshape(*output_shape)
