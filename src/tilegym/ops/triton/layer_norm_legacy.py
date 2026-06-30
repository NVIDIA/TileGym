# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import get_available_triton_backend
from tilegym.backend import register_impl

# Adapted from https://github.com/openai/triton

# =============================================================================
# Utility functions for persistent layer norm
# =============================================================================


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Switch tensor to contiguous layout if needed."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _persistent_layer_norm_pre_hook(nargs):
    """Pre-hook function to set block shapes for host descriptors"""
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]

    # Set block shapes for weight descriptor
    if isinstance(nargs.get("w_desc"), TensorDescriptor):
        nargs["w_desc"].block_shape = [BLOCK_D]

    # Set block shapes for bias descriptor (if present)
    if isinstance(nargs.get("b_desc"), TensorDescriptor):
        nargs["b_desc"].block_shape = [BLOCK_D]

    # Set block shapes for input descriptor
    if isinstance(nargs.get("x_desc"), TensorDescriptor):
        nargs["x_desc"].block_shape = [BLOCK_N, BLOCK_D]


def _persistent_layer_norm_early_config_prune(configs, named_args, **kwargs):
    """Prune configs that exceed register limits."""
    BLOCK_D = kwargs["BLOCK_D"]
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_N = kw["BLOCK_N"]
        if BLOCK_N * BLOCK_D / (8 * 32) <= 256:
            pruned_configs.append(config)
    return pruned_configs


def _get_persistent_layer_norm_autotune_config():
    """Get autotune configurations for persistent layer norm."""
    return [
        triton.Config(dict(BLOCK_N=BN), num_stages=s, pre_hook=_persistent_layer_norm_pre_hook)
        for BN in [2, 4, 8, 16, 32]
        for s in [2, 4, 6, 8]
    ]


@triton.autotune(
    configs=_get_persistent_layer_norm_autotune_config(),
    key=["N", "D", "IS_SWISH", "TRAINING", "COMPUTE_MEAN_AND_RSTD"],
    prune_configs_by={"early_config_prune": _persistent_layer_norm_early_config_prune},
)
@triton.jit
def _persistent_layer_norm_fwd(
    x_desc,
    Y,
    w_desc,
    b_desc,
    Mean,
    Rstd,
    N,
    D,
    eps,
    stride_y,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent layer norm forward kernel with TMA support."""
    pid = tl.program_id(0)
    # Calculate upper bound (equivalent to %upper_bound = arith.divsi %M, %c4_i32)
    upper_bound = tl.cdiv(N, BLOCK_N)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = w_desc.load([0]).to(tl.float32)
    b = b_desc.load([0]).to(tl.float32)

    for current_pid in range(pid, upper_bound, NUM_SMS):
        row_offset = current_pid * BLOCK_N
        rows = row_offset + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mask = row_mask[:, None] & col_mask[None, :]
        x = x_desc.load([row_offset, 0]).to(tl.float32)

        if COMPUTE_MEAN_AND_RSTD:
            # Step 1: Compute x^2
            x_squared = x * x
            # Step 2: Reduce sum along axis=1 (columns) - equivalent to nv_tileaa.reduce
            avg_square = tl.sum(x_squared, axis=1) / D  # Shape: [BLOCK_N]
            mean = tl.sum(x, axis=1) / D  # Shape : [BLOCK_N]
            var = avg_square - mean * mean
            rstd = 1 / tl.sqrt(var + eps)
            if TRAINING:
                if BLOCK_N == 1:
                    tl.store(Mean + rows, mean)
                    tl.store(Rstd + rows, rstd)
                else:
                    tl.store(Mean + rows, mean, mask=row_mask)
                    tl.store(Rstd + rows, rstd, mask=row_mask)
        else:
            if BLOCK_N == 1:
                mean = tl.load(Mean + rows)
                rstd = tl.load(Rstd + rows)
            else:
                mean = tl.load(Mean + rows, mask=row_mask)
                rstd = tl.load(Rstd + rows, mask=row_mask)

        if BLOCK_N != 1:
            mean = mean[:, None]
            rstd = rstd[:, None]
        # Normalize and apply linear transformation
        x_hat = (x - mean) * rstd
        w_broadcasted = w[None, :]
        b_broadcasted = b[None, :]
        y = x_hat * w_broadcasted + b_broadcasted
        if IS_SWISH:
            y = tl.sigmoid(y) * x
        # Write output

        out_ptrs = Y + rows[:, None] * stride_y + cols[None, :]
        tl.store(out_ptrs, y.to(tl.bfloat16), mask=mask)


def _triton_persistent_layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Persistent layer norm forward pass with TMA support.

    Args:
        x: Input tensor of shape (N, D)
        weight: Weight tensor of shape (D,)
        bias: Bias tensor of shape (D,)
        eps: Epsilon for numerical stability
        mean: Optional pre-computed mean tensor
        rstd: Optional pre-computed reciprocal std tensor

    Returns:
        Tuple of (output, mean, rstd, BLOCK_D, num_warps)
    """
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = _switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert bias is not None and weight is not None
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D

    y = torch.empty_like(x)
    compute_mean_and_rstd = mean is None or rstd is None
    if mean is None:
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    if rstd is None:
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    BLOCK_D = triton.next_power_of_2(D)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda meta: (
        min(
            NUM_SMS,
            triton.cdiv(N, meta["BLOCK_N"]) * triton.cdiv(D, meta["BLOCK_D"]),
        ),
    )
    x_desc = TensorDescriptor(x, shape=list(x.shape), strides=list(x.stride()), block_shape=[1, BLOCK_D])
    w_desc = TensorDescriptor(weight, shape=list(weight.shape), strides=list(weight.stride()), block_shape=[BLOCK_D])
    b_desc = TensorDescriptor(bias, shape=list(bias.shape), strides=list(bias.stride()), block_shape=[BLOCK_D])
    # pyre-ignore[28]
    _persistent_layer_norm_fwd[grid](
        x_desc,
        y,
        w_desc,
        b_desc,
        mean,
        rstd,
        N,
        D,
        eps,
        y.stride(0),
        IS_SWISH=False,
        TRAINING=True,
        BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
        NUM_SMS=NUM_SMS,
    )
    num_warps = 8
    return y, mean, rstd, BLOCK_D, num_warps


# =============================================================================
# Legacy Layer Norm Kernels
# =============================================================================


def _get_layer_norm_fwd_fused_configs():
    """Autotune configs for the legacy LN forward kernel (TileIR only).

    Norm-Like kernel: tune occupancy + num_warps. ``num_stages=1`` is held
    fixed because multi-stage pipelining offers little for the small reduction
    loops in this kernel (and regressed measurably during exploration).
    ``occupancy`` is the dominant TileIR knob — packing multiple blocks per
    SM is what unlocks the bulk of the speedup at small/medium N.
    BLOCK_SIZE is computed at launch time from N (next_power_of_2) so it stays
    a constexpr argument and is NOT in the config kwargs.
    """
    return [
        triton.Config({"occupancy": occ}, num_warps=w, num_stages=1)
        for occ in [1, 2, 4, 8, 16, 24, 32]
        for w in [1, 2, 4, 8, 16]
    ]


if get_available_triton_backend() == "nvt":
    _layer_norm_fwd_fused_decorator = triton.autotune(
        configs=_get_layer_norm_fwd_fused_configs(),
        key=["N", "BLOCK_SIZE"],
    )
else:

    def _layer_norm_fwd_fused_decorator(fn):
        return fn


@_layer_norm_fwd_fused_decorator
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # X row stride (last-dim stride assumed to be 1)
    stride_y_row,  # Y row stride (last-dim stride assumed to be 1)
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    weight_shift,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride_y_row
    X += row * stride_x_row
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask) + weight_shift
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, weight_shift=0.0):
        # Reshape input to 2D and normalize layout to what the kernel needs.
        # The kernel reads columns as ``X + cols`` (assumes last-dim stride
        # == 1) and walks rows as ``X += row * stride_x_row``. So we need
        # last-dim unit stride, but the row pitch can be arbitrary and is
        # passed explicitly. ``switch_to_contiguous_if_needed`` copies iff
        # the last-dim stride is not 1, keeping row-sliced views like
        # ``x[:, :N]`` (non-contiguous but ``stride(-1) == 1``) zero-copy.
        x_arg = _switch_to_contiguous_if_needed(x.reshape(-1, x.shape[-1]))
        M, N = x_arg.shape
        # Allocate output as row-major contiguous 2D so that ``y.view(x.shape)``
        # at the end always succeeds (independent of x's layout).
        y_arg = torch.empty((M, N), dtype=x.dtype, device=x.device)
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps for the PTX path, which does not autotune.
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        if get_available_triton_backend() == "nvt":
            # TileIR: autotuner picks num_warps/occupancy.
            _layer_norm_fwd_fused[(M,)](
                x_arg,
                y_arg,
                weight,
                bias,
                mean,
                rstd,
                x_arg.stride(0),
                y_arg.stride(0),
                N,
                eps,
                weight_shift,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            # PTX: keep the original heuristic launch; ``occupancy`` is a
            # no-op constexpr on this backend.
            _layer_norm_fwd_fused[(M,)](
                x_arg,
                y_arg,
                weight,
                bias,
                mean,
                rstd,
                x_arg.stride(0),
                y_arg.stride(0),
                N,
                eps,
                weight_shift,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=1,
            )
        # Restore the original (possibly >2D) shape. ``y_arg`` is contiguous,
        # so ``view`` always succeeds regardless of ``x``'s original layout.
        y = y_arg.view(x.shape)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.weight_shift = weight_shift
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("LayerNorm backward is not implemented for this backend")


@register_impl("layer_norm_legacy", backend="triton")
def layer_norm(input, normalized_shape, weight, bias, eps, weight_shift=0.0, **kwargs):
    r"""
    Returns the LayerNorm of input along dimension N

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Unused
        weight: Tensor of shape (N,)
        bias: Tensor of shape (N,)
        eps: small scaler to be added to
            variance calculation prior to division.
        weight_shift: float value to be added to the weight
        **kwargs: Additional arguments for backend-specific configurations
    """
    return _LayerNorm.apply(input, normalized_shape, weight, bias, eps, weight_shift)


@register_impl("persistent_layer_norm", backend="triton")
def persistent_layer_norm(
    input: torch.Tensor,
    normalized_shape,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    r"""
    Returns the persistent LayerNorm of input with TMA support.

    This is an optimized implementation using TMA descriptors and autotune.

    Args:
        input: Tensor of shape (N, D)
        normalized_shape: Unused (for API compatibility)
        weight: Tensor of shape (D,)
        bias: Tensor of shape (D,)
        eps: Epsilon for numerical stability
        mean: Optional pre-computed mean tensor
        rstd: Optional pre-computed reciprocal std tensor
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Tuple of (output, mean, rstd, BLOCK_D, num_warps)
    """
    # Reshape input to 2D if needed
    original_shape = input.shape
    if input.dim() != 2:
        input = input.reshape(-1, input.shape[-1])

    y, mean_out, rstd_out, block_d, num_warps = _triton_persistent_layer_norm_fwd(input, weight, bias, eps, mean, rstd)

    # Reshape output back to original shape if needed
    if len(original_shape) != 2:
        y = y.reshape(original_shape)

    return y, mean_out, rstd_out, block_d, num_warps
