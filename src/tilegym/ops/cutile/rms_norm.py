# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch
import torch.nn as nn
import math

from tilegym.backend import register_impl

from .utils import next_power_of_2

@ct.kernel
def rms_norm_backward_dx(
    dx,
    dy,
    x,
    weight,
    Rstd,
    TILE_SIZE: ct.Constant[int],
):
    """
    Compute input gradients for RMSNorm backward pass.
    
    Formula: dx_{m,i} = dy_{m,i} w_i / r_m - x_{m,i} / (N r_m^3) * sum_j dy_{m,j} w_j x_{m,j}
    where:
      - dy_{m,i} = dy[m,i] (upstream gradient)
      - w_i = weight[i] (scale parameter)
      - r_m = 1 / rstd[m] (RMS for row m)
      - N = number of columns
    
    See rms_norm_backward_annotated() for detailed derivation.
    
    Each block handles exactly one row and processes all columns at once.
    TILE_SIZE should be >= N (number of columns).
    """
    row_idx = ct.bid(0)
    M, N = x.shape

    # Load entire row from input and gradient
    input_row = ct.load(x, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)
    gradient_row = ct.load(dy, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.ZERO)

    # Load reciprocal std (1D tensor [M]) and reshape for broadcasting
    inv_std_row = ct.load(Rstd, index=(row_idx,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
    inv_std_row = ct.reshape(inv_std_row, (1, 1))  # Reshape to [1, 1] for broadcasting

    # Load weight vector and reshape for broadcasting
    weight_vector = ct.load(weight, index=(0,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    weight_vector = ct.reshape(weight_vector, (1, TILE_SIZE))  # Reshape to [1, TILE_SIZE] for broadcasting
    
    # Compute sum_j dy_{m,j} w_j x_{m,j} for the correction term
    weighted_gradient_product = input_row * weight_vector * gradient_row
    weighted_gradient_sum = ct.sum(weighted_gradient_product, axis=1, keepdims=True)  # [1, 1]

    # Compute normalization correction: x_{m,i} / (N r_m^3) * sum_j dy_{m,j} w_j x_{m,j}
    # Since inv_std_row = 1/r_m, we have r_m^3 = 1/(inv_std_row^3)
    inv_std_cubed = inv_std_row * inv_std_row * inv_std_row  # [1, 1]
    norm_factor = ct.full((1, 1), N * 1.0, dtype=ct.float32)  # [1, 1]
    normalization_correction_coeff = input_row * inv_std_cubed / norm_factor  # [1, TILE_SIZE]
    normalization_correction = normalization_correction_coeff * weighted_gradient_sum  # [1, TILE_SIZE]
    
    # Compute direct term: dy_{m,i} w_i / r_m = gradient_row * weight_vector * inv_std_row
    scaled_gradient = gradient_row * weight_vector * inv_std_row  # [1, TILE_SIZE]
    
    # Final dx: direct term minus normalization correction
    input_gradient_row = scaled_gradient - normalization_correction  # [1, TILE_SIZE]
    
    # Convert back to the original dtype of dx
    input_gradient_row = ct.astype(input_gradient_row, dx.dtype)
    
    # Store the result back to dx
    ct.store(dx, index=(row_idx, 0), tile=input_gradient_row)    

@ct.kernel
def rms_norm_backward_dw(
    dw,
    dy,
    x,
    Rstd,
    TILE_SIZE: ct.Constant[int],
):
    """
    Compute weight gradients for RMSNorm backward pass.
    
    Formula: dw_i = sum_over_rows(dy[m,i] * x[m,i] * rstd[m])
    where rstd[m] = 1/r_m (reciprocal standard deviation for row m)
    
    Each block handles exactly one column and processes all rows at once.
    TILE_SIZE should be M (number of rows).
    """
    col_idx = ct.bid(0)
    M, N = x.shape
    
    # Load entire column from input and gradient
    input_col = ct.load(x, index=(0, col_idx), shape=(TILE_SIZE, 1), padding_mode=ct.PaddingMode.ZERO)
    gradient_col = ct.load(dy, index=(0, col_idx), shape=(TILE_SIZE, 1), padding_mode=ct.PaddingMode.ZERO)
    
    # Load reciprocal std (1D tensor [M]) and reshape for broadcasting
    inv_std_col = ct.load(Rstd, index=(0,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    inv_std_col = ct.reshape(inv_std_col, (TILE_SIZE, 1))  # Reshape to [TILE_SIZE, 1] for broadcasting
    
    # Convert to float32 for stable accumulation (matches annotated backward function)
    input_col = ct.astype(input_col, ct.float32)
    gradient_col = ct.astype(gradient_col, ct.float32)
    inv_std_col = ct.astype(inv_std_col, ct.float32)
    
    # Compute element-wise: dy[m,i] * x[m,i] * rstd[m] for all rows
    column_contributions = gradient_col * input_col * inv_std_col  # [TILE_SIZE, 1]
    
    # Sum over all rows to get the weight gradient for this column
    weight_gradient = ct.sum(column_contributions, axis=0, keepdims=False)  # [1]
    weight_gradient = ct.reshape(weight_gradient, (1,))  # Ensure it's [1]
    
    # Convert back to the original dtype of dw
    weight_gradient = ct.astype(weight_gradient, dw.dtype)
    
    # Store the result back to dw
    ct.store(dw, index=(col_idx,), tile=weight_gradient)

def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    dx = dx.detach()
    dw = dw.detach()

    TILE_SIZE_N = next_power_of_2(N)
    TILE_SIZE_M = next_power_of_2(M)
    
    # Kernel 1: dx (row-parallel)
    grid_dx = (M,)
    ct.launch(
        torch.cuda.current_stream(),
        grid_dx,
        rms_norm_backward_dx,
        (dx, dy, x, weight, rstd, TILE_SIZE_N),
    )

    # Kernel 2: dw (column-parallel)
    grid_dw = (N,)
    ct.launch(
        torch.cuda.current_stream(),
        grid_dw,
        rms_norm_backward_dw,
        (dw, dy, x, rstd, TILE_SIZE_M),
    )

    # Reshape dx back, dw already correct
    return dx.view(*x_shape), dw


@ct.kernel
def rms_norm_kernel_gather(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Standard RMSNorm kernel for non-static persistent mode with ptr loads"""
    row = ct.bid(0) # 
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
        # xj = ct.load(x, index=(row, j * TILE_SIZE), shape=(1, TILE_SIZE), latency=1) # TODO: Test this out
        xj = ct.astype(xj, ct.float32)
        _rms += xj * xj

    # Calculate RMS Norm
    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(Rstd, row, rms)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, latency=1)
        wj = ct.astype(wj, ct.float32)
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, ct.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


@ct.kernel
def rms_norm_kernel_static_persistent(
    X,  # Input tensor
    Y,  # Output tensor
    W,  # Weight tensor
    TILE_SIZE_M: ct.Constant[int],  # rows per tile
    TILE_SIZE_N: ct.Constant[int],  # columns per tile
    eps: ct.Constant[float],  # Epsilon value
):
    """
    CuTile static persistent RMSNorm kernel that uses a persistent approach,
    where NUM_SMS tile blocks are launched and each tile block processes multiple output tiles
    for better efficiency.
    """
    # Get program ID
    bid = ct.bid(0)

    # Infer tensor dimensions from input shape
    M = X.shape[0]  # Number of rows
    N = X.shape[1]  # Number of columns

    # Calculate upper bound
    upper_bound = (M + TILE_SIZE_M - 1) // TILE_SIZE_M

    # Load weight vector once (shared across all tiles processed by this program)
    w = ct.load(W, index=(0,), shape=(TILE_SIZE_N,))
    w = ct.astype(w, ct.float32)

    # Static persistent loop: each  processes multiple tiles
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        # Load input tile
        x = ct.load(
            X,
            index=(current_bid, 0),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            latency=10,  # +2% perf from this hint
        )
        x = ct.astype(x, ct.float32)

        # Step 1: Compute x^2
        x_squared = ct.mul(x, x)

        # Step 2: Reduce sum along axis=1 (columns)
        x2_sum = ct.sum(x_squared, axis=1, keepdims=True)  # Shape: [TILE_SIZE_M, 1]

        # Step 3: Compute variance (divide by N)
        N_f32 = ct.full((TILE_SIZE_M, 1), N * 1.0, dtype=ct.float32)
        variance = ct.truediv(x2_sum, N_f32)

        # Step 4: Add epsilon and compute rsqrt
        eps_tensor = ct.full((TILE_SIZE_M, 1), eps, dtype=ct.float32)
        variance_eps = ct.add(variance, eps_tensor)
        rsqrt_var = ct.rsqrt(variance_eps)

        # Step 5: Apply normalization
        x_normalized = ct.mul(x, rsqrt_var)

        # Step 6: Apply linear transformation
        # Broadcast weight to match input shape
        w_broadcasted = ct.reshape(w, (1, TILE_SIZE_N))

        # Apply linear transformation: y = x_normalized * w
        y = ct.mul(x_normalized, w_broadcasted)

        # Convert back to original dtype
        y = ct.astype(y, X.dtype)

        # Store result
        ct.store(
            Y,
            index=(current_bid, 0),
            tile=y,
            allow_tma=False,  # +30% perf
            latency=3,  # +3% perf from this hint
        )


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        eps,
        bias=None,
        static_persistent=None,
    ):
        """
        Unified RMSNorm forward pass supporting both standard and static persistent modes.

        Args:
            x: Input tensor of shape [M, N]
            normalized_shape: Normalization shape (for compatibility, not used)
            weight: Weight tensor of shape [N]
            eps: Epsilon value for numerical stability
            bias: Bias tensor of shape [N], default is None
            static_persistent: Whether to use static persistent kernel, default is False

        Returns:
            Normalized and transformed tensor of same shape as input
        """
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        # Reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])

        # Allocate output tensor
        y = torch.empty_like(x_arg)
        M, N = x_arg.shape
        y = y.detach()
        weight = weight.detach()
        if bias is not None:
            bias = bias.detach()
        x_arg = x_arg.detach()

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        if static_persistent is None:
            if M > NUM_SMS * 2:
                # Heuristic for static persistent mode: if we need run over 2 waves, use static persistent mode
                static_persistent = True
            else:
                static_persistent = False

        if static_persistent:
            # Static persistent mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in static persistent CuTile RMSNorm")

            def ceil_div(a, b):
                return (a + b - 1) // b

            TILE_SIZE_M = 4  # Default value, could be made configurable
            TILE_SIZE_N = next_power_of_2(N)

            # Other block sizes are more optimal when other dimension is too large/too small
            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2

            grid_size = min(
                NUM_SMS,
                ceil_div(M, TILE_SIZE_M) * ceil_div(N, TILE_SIZE_N),
            )
            grid = (grid_size,)
            kernel_sp = rms_norm_kernel_static_persistent
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                kernel_sp,
                (x_arg, y, weight, TILE_SIZE_M, TILE_SIZE_N, eps),
            )
        else:
            # Standard mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")

            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
            MAX_FUSED_SIZE = 4096 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            kernel = rms_norm_kernel_gather
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                kernel,
                (
                    x_arg,
                    weight,
                    y,
                    rstd,
                    N,
                    eps,
                    TILE_SIZE,
                ),
            )

            # Save variables needed for backward pass
            ctx.save_for_backward(x, weight, rstd)
            ctx.TILE_SIZE = TILE_SIZE
            ctx.eps = eps

        return y.view(*x.shape)

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for RMSNorm.
        Retrieves saved tensors and delegates to rms_norm_backward().
        """
        x, weight, rstd = ctx.saved_tensors
        
        # Call the standalone backward function
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        
        # Return gradients: (x, normalized_shape, weight, eps, bias, static_persistent)
        return dx, None, dw, None, None, None


@register_impl("rms_norm", backend="cutile")
def rms_norm(input, normalized_shape, weight, eps, bias=None, static_persistent=None, **kwargs):
    """
    Root mean square normalization implemented using CUDA Tile

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation
        bias: Bias tensor of shape (N,), default is None (not supported in cutile)
        static_persistent: Whether to use static persistent kernel, default is False
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    return RMSNorm.apply(input, normalized_shape, weight, eps, bias, static_persistent)


class TileRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm implementation using CUDA Tile

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon value for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states, static_persistent=None):
        """
        Forward pass with optional static_persistent override

        Args:
            hidden_states: Input tensor
            static_persistent: Default is None, which means use heuristic to
                               decide whether to use static persistent mode for better performance
        """
        return rms_norm(
            hidden_states,
            None,
            self.weight,
            self.variance_epsilon,
            static_persistent=static_persistent,
        )

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for RMSNorm.
        Retrieves saved tensors and delegates to rms_norm_backward().
        """
        x, weight, rstd = ctx.saved_tensors
        
        # Call the standalone backward function
        dx, dw = rms_norm_backward(x, dy, weight, rstd)
        
        # Return gradients: (x, normalized_shape, weight, eps, bias, static_persistent)
        return dx, None, dw, None, None, None

    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@register_impl("get_rms_norm_module", backend="cutile")
def get_rms_norm_module():
    return TileRMSNorm
