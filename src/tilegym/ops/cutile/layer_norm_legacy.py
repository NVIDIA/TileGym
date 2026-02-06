# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride: ct.Constant[int],
    N: ct.Constant[int],
    eps: ct.Constant[float],
    weight_shift: ct.Constant[float],
    BLOCK_SIZE: ct.Constant[int],
):
    # Map the program id to the row of X and Y it should compute.
    row = ct.bid(0)

    # Compute mean
    _mean = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        mask = ct.less(cols, N)
        # Calculate the offset for the current row and column
        offset = ct.add(ct.mul(row, stride), cols)
        a = ct.gather(X, offset, padding_value=0)
        a = ct.astype(a, ct.float32)
        _mean = ct.add(_mean, a)
    mean = ct.truediv(ct.sum(_mean, axis=0), N)

    # Compute variance
    _var = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        mask = ct.less(cols, N)
        # Calculate the offset for the current row and column
        offset = ct.add(ct.mul(row, stride), cols)
        x = ct.gather(X, offset, padding_value=0)
        x = ct.astype(x, ct.float32)
        x = ct.where(mask, ct.sub(x, mean), ct.zeros((BLOCK_SIZE,), dtype=ct.float32))
        _var = ct.add(_var, ct.mul(x, x))
    var = ct.truediv(ct.sum(_var, axis=0), N)
    rstd = ct.rsqrt(ct.add(var, eps))

    # Write mean / rstd
    mean_offset = ct.full((1,), row, dtype=ct.int32)
    mean_val_reshaped = ct.reshape(mean, (1,))
    ct.scatter(Mean, mean_offset, mean_val_reshaped)

    rstd_offset = ct.full((1,), row, dtype=ct.int32)
    rstd_val_reshaped = ct.reshape(rstd, (1,))
    ct.scatter(Rstd, rstd_offset, rstd_val_reshaped)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        mask = ct.less(cols, N)
        w = ct.gather(W, cols, padding_value=0)
        w = ct.add(w, weight_shift)
        b = ct.gather(B, cols, padding_value=0)
        # Calculate the offset for the current row and column
        offset = ct.add(ct.mul(row, stride), cols)
        x = ct.gather(X, offset, padding_value=0)
        x = ct.astype(x, ct.float32)
        x_hat = ct.mul(ct.sub(x, mean), rstd)
        y = ct.add(ct.mul(x_hat, w), b)
        y = ct.astype(y, X.dtype)
        # Calculate the output offset
        y_offset = ct.add(ct.mul(row, stride), cols)
        ct.scatter(Y, y_offset, y)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, weight_shift=0.0):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, 1 << (N - 1).bit_length())
        assert N % BLOCK_SIZE == 0, (
            f"N % BLOCK_SIZE == {N % BLOCK_SIZE}, expected 0, otherwise the kernel is not supported yet"
        )
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Flatten tensors to 1D for gather/scatter operations
        x_arg_flat = x_arg.reshape(-1)
        y_flat = y.reshape(-1)

        # enqueue kernel
        grid = (M,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _layer_norm_fwd_kernel,
            (
                x_arg_flat,
                y_flat,
                weight,
                bias,
                mean,
                rstd,
                x_arg.stride(0),
                N,
                eps,
                weight_shift,
                BLOCK_SIZE,
            ),
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.weight_shift = weight_shift
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("LayerNorm backward is not implemented for this backend")


@register_impl("layer_norm_legacy", backend="cutile")
def layer_norm_legacy(input, normalized_shape, weight, bias, eps, weight_shift=0.0, **kwargs):
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
    return LayerNorm.apply(input, normalized_shape, weight, bias, eps, weight_shift)


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Switch tensor to contiguous layout if needed."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _persistent_layer_norm_autotune_configs():
    """
    Autotune config generator for persistent layer norm.

    Generates configurations:
    - BLOCK_N: [2, 4, 8, 16, 32] - number of rows per block
    - num_ctas: [1] - single CTA for this kernel
    """
    # BLOCK_N options
    block_n_options = [2, 4, 8, 16, 32]

    # num_ctas options
    num_ctas_options = [1]

    for block_n in block_n_options:
        for num_ctas in num_ctas_options:
            yield SimpleNamespace(
                BLOCK_N=block_n,
                num_ctas=num_ctas,
            )


def _get_default_persistent_layer_norm_configs():
    """GPU-specific defaults when autotune is disabled."""
    return {
        "BLOCK_N": 8,
        "num_ctas": 1,
    }


def _persistent_layer_norm_early_config_prune(configs, N, D, BLOCK_D):
    """Prune configs that exceed register limits."""
    pruned_configs = []
    for cfg in configs:
        BLOCK_N = cfg.BLOCK_N
        # Register limit check: BLOCK_N * BLOCK_D / (8 * 32) <= 256
        if BLOCK_N * BLOCK_D / (8 * 32) <= 256:
            pruned_configs.append(cfg)
    return pruned_configs


@ct.kernel
def _persistent_layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    N: ct.Constant[int],
    D: ct.Constant[int],
    eps: ct.Constant[float],
    stride_x: ct.Constant[int],
    stride_y: ct.Constant[int],
    IS_SWISH: ct.Constant[int],
    TRAINING: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
    COMPUTE_MEAN_AND_RSTD: ct.Constant[int],
    NUM_SMS: ct.Constant[int],
):
    """Persistent layer norm forward kernel with cuTile."""
    pid = ct.bid(0)
    # Calculate upper bound
    upper_bound = ct.cdiv(N, BLOCK_N)

    cols = ct.arange(BLOCK_D, dtype=ct.int32)
    col_mask = cols < D

    # Load weights (1D)
    # TMA handles padding (defaults to 0) automatically when BLOCK_D > D
    w = ct.load(W, index=(0,), shape=(BLOCK_D,), padding_mode=PAD_ZERO)
    b = ct.load(B, index=(0,), shape=(BLOCK_D,), padding_mode=PAD_ZERO)

    # Cast to float32
    w = ct.astype(w, ct.float32)
    b = ct.astype(b, ct.float32)

    # Persistent Grid Stride Loop
    for current_pid in range(pid, upper_bound, NUM_SMS):
        row_offset = current_pid * BLOCK_N
        rows = row_offset + ct.arange(BLOCK_N, dtype=ct.int32)
        row_mask = rows < N
        mask = row_mask[:, None] & col_mask[None, :]

        x_tile = ct.load(X, index=(current_pid, 0), shape=(BLOCK_N, BLOCK_D), padding_mode=PAD_ZERO, latency=4)
        x = ct.astype(x_tile, ct.float32)

        if COMPUTE_MEAN_AND_RSTD:
            # Step 1: Compute x^2
            x_squared = x * x
            avg_square = ct.sum(x_squared, axis=1) / D
            mean = ct.sum(x, axis=1) / D
            var = avg_square - mean * mean
            rstd = ct.rsqrt(var + eps)
            if TRAINING:
                ct.store(Mean, index=(current_pid,), tile=mean, allow_tma=False)
                ct.store(Rstd, index=(current_pid,), tile=rstd, allow_tma=False)
        else:
            mean = ct.gather(Mean, rows)
            rstd = ct.gather(Rstd, rows)

        if BLOCK_N != 1:
            mean = mean[:, None]
            rstd = rstd[:, None]

        # Normalize and apply linear transformation
        x_hat = (x - mean) * rstd
        w_broadcasted = w[None, :]
        b_broadcasted = b[None, :]
        y = x_hat * w_broadcasted + b_broadcasted

        if IS_SWISH:
            y = ct.sigmoid(y) * x
        y = ct.astype(y, ct.bfloat16)
        ct.store(Y, index=(current_pid, 0), tile=y, allow_tma=False)


def _persistent_layer_norm_autotune_base(
    stream,
    x,
    y,
    weight,
    bias,
    mean,
    rstd,
    N,
    D,
    eps,
    stride_x,
    stride_y,
    IS_SWISH,
    TRAINING,
    BLOCK_D,
    COMPUTE_MEAN_AND_RSTD,
):
    """
    Autotuned kernel launch for persistent layer norm.
    """
    NUM_SM = torch.cuda.get_device_properties(x.device).multi_processor_count

    # Prune configs based on register limits
    all_configs = list(_persistent_layer_norm_autotune_configs())
    pruned_configs = _persistent_layer_norm_early_config_prune(all_configs, N, D, BLOCK_D)

    # If all configs pruned, use default
    if not pruned_configs:
        pruned_configs = [SimpleNamespace(**_get_default_persistent_layer_norm_configs())]

    def search_space():
        for cfg in pruned_configs:
            yield cfg

    def args_fn(cfg):
        BLOCK_N = cfg.BLOCK_N
        num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
        grid_size = min(NUM_SM, num_row_blocks)
        return (
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            D,
            eps,
            stride_x,
            stride_y,
            IS_SWISH,
            TRAINING,
            BLOCK_N,
            BLOCK_D,
            COMPUTE_MEAN_AND_RSTD,
            grid_size,  # NUM_SMS
        )

    def grid_fn(cfg):
        BLOCK_N = cfg.BLOCK_N
        num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
        grid_size = min(NUM_SM, num_row_blocks)
        return (grid_size, 1, 1)

    ct_experimental.autotune_launch(
        stream,
        grid_fn=grid_fn,
        kernel=_persistent_layer_norm_fwd_kernel,
        args_fn=args_fn,
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
        },
        search_space=search_space,
    )


def cutile_persistent_layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
):
    """
    Persistent layer norm forward pass with cuTile and autotune support.

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
    x = switch_to_contiguous_if_needed(x)
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

    # Calculate block sizes
    BLOCK_D = next_power_of_2(D)

    # Check if autotune is enabled (default: enabled)
    enable_autotune = os.environ.get("DISABLE_CUTILE_TUNE", "0") != "1"

    if enable_autotune:
        _persistent_layer_norm_autotune_base(
            torch.cuda.current_stream(),
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            0,  # IS_SWISH
            1,  # TRAINING
            BLOCK_D,
            1 if compute_mean_and_rstd else 0,
        )
    else:
        # Use fixed default configs
        configs = _get_default_persistent_layer_norm_configs()
        BLOCK_N = configs["BLOCK_N"]
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
        grid_size = min(NUM_SMS, num_row_blocks)
        grid = (grid_size, 1, 1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _persistent_layer_norm_fwd_kernel,
            (
                x,
                y,
                weight,
                bias,
                mean,
                rstd,
                N,
                D,
                eps,
                x.stride(0),
                y.stride(0),
                0,  # IS_SWISH
                1,  # TRAINING
                BLOCK_N,
                BLOCK_D,
                1 if compute_mean_and_rstd else 0,
                grid_size,  # NUM_SMS
            ),
        )

    num_warps = 8
    return y, mean, rstd, BLOCK_D, num_warps


@register_impl("persistent_layer_norm", backend="cutile")
def persistent_layer_norm(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    **kwargs,
):
    r"""
    Returns the persistent LayerNorm of input with cuTile.

    This is an optimized implementation using persistent kernel pattern.

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

    y, mean_out, rstd_out, block_d, num_warps = cutile_persistent_layer_norm_fwd(input, weight, bias, eps, mean, rstd)

    # Reshape output back to original shape if needed
    if len(original_shape) != 2:
        y = y.reshape(original_shape)

    return y, mean_out, rstd_out, block_d, num_warps
