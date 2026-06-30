# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import get_available_triton_backend
from tilegym.backend import get_current_backend
from tilegym.backend import register_impl

backend = get_available_triton_backend()


def _rms_norm_pre_hook(nargs):
    """Pre-hook function to set block shapes for host descriptors"""
    BLOCK_SIZE_M = nargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = nargs["BLOCK_SIZE_N"]

    # Set block shapes for weight descriptor
    if isinstance(nargs.get("w_desc"), TensorDescriptor):
        nargs["w_desc"].block_shape = [BLOCK_SIZE_N]

    # Set block shapes for bias descriptor (if present)
    if isinstance(nargs.get("b_ptr"), TensorDescriptor):
        nargs["b_ptr"].block_shape = [BLOCK_SIZE_N]

    # Set block shapes for input descriptor
    if isinstance(nargs.get("X"), TensorDescriptor):
        nargs["X"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]


def _early_config_prune(configs, named_args, **kwargs):
    BLOCK_SIZE_N = kwargs["BLOCK_SIZE_N"]
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_SIZE_M = kw["BLOCK_SIZE_M"]
        num_warps = config.num_warps
        if BLOCK_SIZE_N * BLOCK_SIZE_M / (num_warps * 32) <= 256:
            pruned_configs.append(config)
    return pruned_configs


def _get_rms_norm_autotune_config():
    """Get autotune configurations for RMS Norm kernel"""
    return [
        triton.Config(dict(BLOCK_SIZE_M=BM), num_stages=s, num_warps=w, pre_hook=_rms_norm_pre_hook)
        for BM in [2, 4, 8, 16]
        for s in [2, 4, 6, 8]
        for w in [4, 8]
    ]


@triton.jit
def _rms_norm_kernel(
    X,  # input tensor
    W,  # weight tensor
    Y,  # output tensor
    Rstd,  # 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
    offset: tl.constexpr,  # offset to add to weight (for Gemma3: offset=1.0)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Standard RMSNorm kernel for non-static persistent mode

    Formula: y_i = (x_i / RMS) * (offset + w_i)
    where RMS = sqrt(sum(x_i^2) / N + eps)
    """
    row = tl.program_id(0)
    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    Y += row * stride
    X += row * stride

    for j in range(0, N, BLOCK_SIZE):
        cols = j + tl.arange(0, BLOCK_SIZE)
        xj = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += xj * xj

    # Calculate RMS Norm
    rms = tl.math.rsqrt(tl.sum(_rms, axis=0) / N + eps)
    tl.store(Rstd + row, rms)

    # Normalize and apply linear transformation with offset
    for j in range(0, N, BLOCK_SIZE):
        cols = j + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        wj = tl.load(W + cols, mask=mask).to(tl.float32)
        xj = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        # Apply offset: y = x_normalized * (offset + w)
        yj = xj * rms * (offset + wj)
        # Write output
        tl.store(Y + cols, yj.to(Y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=_get_rms_norm_autotune_config(),
    key=["M", "N", "USE_BIAS"],
    prune_configs_by={"early_config_prune": _early_config_prune},
)
@triton.jit
def _rms_norm_kernel_static_persistent(
    X,  # input tensor
    Y,  # output tensor
    w_desc,  # weight tensor
    b_ptr,  # bias tensor
    N,  # number of columns
    M,  # number of rows
    stride_m,  # row stride
    eps,  # epsilon value
    offset: tl.constexpr,  # offset value
    BLOCK_SIZE_M: tl.constexpr,  # rows per block
    BLOCK_SIZE_N: tl.constexpr,  # columns per block
    USE_BIAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Triton static persistent RMSNorm kernel that processes multiple blocks per program.
    Each program processes multiple blocks in a loop for better efficiency.

    Formula: y_i = (x_i / RMS) * (offset + w_i) + b_i
    where RMS = sqrt(sum(x_i^2) / N + eps)
    """
    # Get program ID
    pid = tl.program_id(0)

    # Calculate upper bound - number of row blocks to process
    upper_bound = tl.cdiv(M, BLOCK_SIZE_M)

    # Load weight and bias vectors once (shared across all blocks processed by this program)
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N
    w = w_desc.load([0]).to(tl.float32)
    if USE_BIAS:
        b_desc = b_ptr
        b = b_desc.load([0]).to(tl.float32)

    # Static persistent loop: each program processes multiple blocks
    for current_block in range(pid, upper_bound, NUM_SMS):
        # Calculate which rows this iteration handles
        row_start = current_block * BLOCK_SIZE_M
        rows = row_start + tl.arange(0, BLOCK_SIZE_M)

        # Create masks
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        # Create tensor descriptor for the input matrix X
        x_desc = X
        # Load a BLOCK_SIZE_M x BLOCK_SIZE_N tile from X
        row_offset = current_block * BLOCK_SIZE_M
        x = x_desc.load([row_offset, 0]).to(tl.float32)

        # Step 1: Compute x^2
        x_squared = x * x

        # Step 2: Reduce sum along axis=1 (columns)
        x2_sum = tl.sum(x_squared, axis=1)  # Shape: [BLOCK_SIZE_M]

        # Step 3: Compute variance (divide by N)
        N_f32 = N.to(tl.float32)
        variance = x2_sum / N_f32  # Shape: [BLOCK_SIZE_M]

        # Step 4: Add epsilon and compute rsqrt
        variance_eps = variance + eps
        rsqrt_var = tl.math.rsqrt(variance_eps)  # Shape: [BLOCK_SIZE_M]

        # Step 5: Broadcast rsqrt to match input shape
        rsqrt_broadcasted = rsqrt_var[:, None]  # Shape: [BLOCK_SIZE_M, 1]

        # Step 6: Apply normalization and linear transformation with offset
        # Normalize: x * rsqrt(variance + eps)
        x_normalized = x * rsqrt_broadcasted

        # Broadcast weight and bias to match input shape
        w_broadcasted = w[None, :]  # Shape: [1, BLOCK_SIZE_N] -> broadcast to [BLOCK_SIZE_M, BLOCK_SIZE_N]
        if USE_BIAS:
            b_broadcasted = b[None, :]  # Shape: [1, BLOCK_SIZE_N] -> broadcast to [BLOCK_SIZE_M, BLOCK_SIZE_N]
        else:
            b_broadcasted = 0.0

        # Apply linear transformation with offset: y = x_normalized * (offset + w) + b
        y = x_normalized * (offset + w_broadcasted) + b_broadcasted

        # Convert back to original dtype
        y = y.to(Y.dtype.element_ty)

        # Store result
        Ys = Y + rows[:, None] * stride_m + cols[None, :]
        tl.store(Ys, y, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        eps,
        bias=None,
        mode=None,
        offset=0.0,
    ):
        """
        Unified Triton RMSNorm forward pass.

        Args:
            x: Input tensor of shape [M, N]
            normalized_shape: Normalization shape (for compatibility, not used)
            weight: Weight tensor of shape [N]
            eps: Epsilon value for numerical stability
            bias: Bias tensor of shape [N], default is None
            mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload").
                  ``multi_wave_cached`` is not implemented in the triton backend
                  and raises NotImplementedError.
            offset: Offset to add to weight (default 0.0 for Llama, 1.0 for Gemma3)

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
        y = torch.empty_like(x)
        M, N = x_arg.shape

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        if mode is None:
            if M > NUM_SMS * 2:
                # heuristic: if we need to run over 2 waves, use static persistent mode
                mode = "static_persistent"
            else:
                mode = "multi_wave_reload"

        if mode == "static_persistent":
            # Static persistent mode
            # TMA descriptors require a global memory allocation
            def alloc_fn(size: int, alignment: int, stream: Optional[int]):
                return torch.empty(size, device="cuda", dtype=torch.int8)

            triton.set_allocator(alloc_fn)
            dummy_block = [1, 1]
            desc_x = TensorDescriptor(x_arg, x_arg.shape, x_arg.stride(), dummy_block)
            desc_weight = TensorDescriptor(weight, weight.shape, weight.stride(), [1])
            desc_bias = TensorDescriptor(bias, bias.shape, bias.stride(), [1]) if bias is not None else None

            def ceil_div(a, b):
                return (a + b - 1) // b

            # Calculate number of tiles - use static value optimized for efficiency
            grid = lambda meta: (
                min(
                    NUM_SMS,
                    ceil_div(M, meta["BLOCK_SIZE_M"]) * ceil_div(N, meta["BLOCK_SIZE_N"]),
                ),
            )

            BLOCK_SIZE_N = triton.next_power_of_2(N)

            _rms_norm_kernel_static_persistent[grid](
                X=desc_x,
                Y=y,
                w_desc=desc_weight,
                b_ptr=desc_bias,
                N=N,
                M=M,
                stride_m=x_arg.stride(0),
                eps=eps,
                offset=offset,
                USE_BIAS=bias is not None,
                NUM_SMS=NUM_SMS,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )

            # Save for backward pass (if needed in the future)
            ctx.save_for_backward(x, weight, bias)
            ctx.eps = eps
            ctx.mode = mode
        elif mode == "multi_wave_reload":
            # Standard multi-wave reload mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in standard Triton RMSNorm")

            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
            MAX_FUSED_SIZE = 4096 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
            _rms_norm_kernel[(M,)](
                x_arg,
                weight,
                y,
                rstd,
                x_arg.stride(0),
                N,
                eps,
                offset,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=1,
            )
            ctx.save_for_backward(x, weight, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.eps = eps
            ctx.offset = offset
            ctx.mode = mode
        elif mode == "multi_wave_cached":
            raise NotImplementedError("multi_wave_cached mode is not implemented for the triton backend")
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: None, 'static_persistent', "
                f"'multi_wave_reload', 'multi_wave_cached'"
            )

        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("RMSNorm backward is not implemented for this backend")


@register_impl("rms_norm", backend="triton")
def rms_norm(input, normalized_shape, weight, eps, bias=None, mode=None, offset=0.0, **kwargs):
    """
    Root mean square normalization implemented using Triton

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation prior to division
        bias: Bias tensor of shape (N,), default is None
        mode: Kernel selection mode (None, "static_persistent", "multi_wave_reload").
              ``multi_wave_cached`` is not implemented in the triton backend
              and raises NotImplementedError.
        offset: Offset to add to weight (default 0.0 for Llama, 1.0 for Gemma3)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    return _RMSNorm.apply(input, normalized_shape, weight, eps, bias, mode, offset)


class _TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=0.0):
        """
        RMSNorm implementation using Triton kernels for faster computation

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon value for numerical stability
            offset: Offset value (default: 0.0 for standard RMSNorm, 1.0 for Gemma3)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.offset = offset

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
            offset=self.offset,
        )

    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.offset + self.weight) * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


class _TritonRMSNormForGemma3(_TritonRMSNorm):
    """
    RMSNorm implementation for Gemma3 models.

    Gemma3 uses 'dim' parameter name instead of 'hidden_size', and initializes
    weights with zeros instead of ones, with offset=1.0.
    """

    def __init__(self, dim, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        """
        RMSNorm implementation for Gemma3 using Triton kernels

        Args:
            dim: Size of the hidden dimension (Gemma3 uses 'dim' instead of 'hidden_size')
            eps: Epsilon value for numerical stability
            offset: Offset value for Gemma3 (default: 1.0), applied dynamically in kernel
            casting_mode: Casting mode for Gemma3 (default: "gemma") - currently not used
            init_fn: Initialization function (default: "zeros") - currently not used
            in_place: Whether to perform operation in-place (default: False) - currently not used
        """
        # Initialize parent with offset
        super().__init__(hidden_size=dim, eps=eps, offset=offset)
        # Override weight initialization to zeros for Gemma3
        self.weight = nn.Parameter(torch.zeros(dim))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


@register_impl("get_rms_norm_module", backend="triton")
def get_rms_norm_module(model: str = "llama"):
    if model == "gemma3":
        return _TritonRMSNormForGemma3
    else:
        return _TritonRMSNorm
