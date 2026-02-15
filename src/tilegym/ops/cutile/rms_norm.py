# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch
import torch.nn as nn

from tilegym.backend import register_impl

from .utils import next_power_of_2


@ct.kernel
def rms_norm_kernel_gather(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    offset: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """
    Standard RMSNorm kernel for non-static persistent mode with ptr loads

    Formula: y = norm(x) * (offset + w)
    For Llama: offset=0.0, For Gemma3: offset=1.0
    """
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=ct.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
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
        # Apply offset: y = x_normalized * (offset + w)
        yj = xj * rms * (offset + wj)
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


@ct.kernel
def rms_norm_kernel_static_persistent(
    X,  # Input tensor
    Y,  # Output tensor
    W,  # Weight tensor
    Rstd,  # rstd output (for backward)
    TILE_SIZE_M: ct.Constant[int],  # rows per tile
    TILE_SIZE_N: ct.Constant[int],  # columns per tile
    eps: ct.Constant[float],  # Epsilon value
    offset: ct.Constant[float],  # Offset value
):
    """
    CuTile static persistent RMSNorm kernel that uses a persistent approach,
    where NUM_SMS tile blocks are launched and each tile block processes multiple output tiles
    for better efficiency.

    Formula: y = norm(x) * (offset + w)
    For Llama: offset=0.0, For Gemma3: offset=1.0
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

        # Store rstd for backward pass
        ct.store(Rstd, index=(current_bid,), tile=ct.reshape(rsqrt_var, (TILE_SIZE_M,)), allow_tma=False)

        # Step 5: Apply normalization
        x_normalized = ct.mul(x, rsqrt_var)

        # Step 6: Apply linear transformation with offset
        # Broadcast weight to match input shape
        w_broadcasted = ct.reshape(w, (1, TILE_SIZE_N))

        # Apply offset to weight: (offset + w)
        offset_tensor = ct.full((1, TILE_SIZE_N), offset, dtype=ct.float32)
        w_with_offset = ct.add(offset_tensor, w_broadcasted)

        # Apply linear transformation: y = x_normalized * (offset + w)
        y = ct.mul(x_normalized, w_with_offset)

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


@ct.kernel(occupancy=1)
def _rms_bwd(dx, dy, x, weight, Rstd, dw_partial, TILE_M: ct.Constant[int], TILE_N: ct.Constant[int]):
    """
    Persistent RMSNorm backward — grid-stride loop with fused dw accumulation.

    Each block accumulates its dw contribution into a (grid, TILE_N) partial
    sum buffer, avoiding the old M×N temp_buffer allocation.

    Only supports offset=0 (Gemma3 backward is not supported).
    """
    bid = ct.bid(0)
    M, N = x.shape[0], x.shape[1]
    blocks = ct.num_blocks(0)
    upper = (M + TILE_M - 1) // TILE_M

    w = ct.astype(ct.load(weight, index=(0,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO), ct.float32)
    w = ct.reshape(w, (1, TILE_N))
    rcp = ct.full((TILE_M, 1), 1.0 / N, dtype=ct.float32)
    dw_acc = ct.full((1, TILE_N), 0.0, dtype=ct.float32)

    for i in range(bid, upper, blocks):
        xt = ct.astype(
            ct.load(x, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO, latency=10),
            ct.float32,
        )
        dyt = ct.astype(
            ct.load(dy, index=(i, 0), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO, latency=10),
            ct.float32,
        )
        r = ct.reshape(
            ct.load(Rstd, index=(i,), shape=(TILE_M,), padding_mode=ct.PaddingMode.ZERO),
            (TILE_M, 1),
        )
        xhat = xt * r
        wdy = dyt * w
        c = ct.sum(xhat * wdy, axis=1, keepdims=True) * rcp
        ct.store(dx, index=(i, 0), tile=ct.astype((wdy - xhat * c) * r, dx.dtype), allow_tma=False, latency=3)
        dw_acc = dw_acc + ct.sum(dyt * xhat, axis=0, keepdims=True)

    ct.store(dw_partial, index=(bid, 0), tile=dw_acc, allow_tma=False)


_bwd_cfg: dict = {}  # (M, N) → (tile_m, tile_n, grid, N)


def _bwd_tiles(M, N):
    """Heuristic tile configuration for backward kernel."""
    T = next_power_of_2(N)
    if T > 4096:
        tm = 1
    elif T <= 2048 or (M >= 8192 and T <= 4096):
        tm = 4
    else:
        tm = 1
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    tiles = (M + tm - 1) // tm
    g = min(NUM_SMS, tiles)
    if tiles <= 64:
        g = min(g, 32)
    return (tm, T, g, N)


def rms_norm_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standalone backward pass using persistent CuTile kernel."""
    x = x.contiguous()
    dy = dy.contiguous()
    weight = weight.contiguous()
    rstd = rstd.contiguous()

    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    M, N = x.shape

    cfg = _bwd_cfg.get((M, N))
    if cfg is None:
        cfg = _bwd_tiles(M, N)
        _bwd_cfg[(M, N)] = cfg
    tm, T, g, No = cfg

    stream = torch.cuda.current_stream()

    dx = torch.empty_like(x)
    dwp = torch.empty((g, T), device=x.device, dtype=torch.float32)
    ct.launch(stream, (g,), _rms_bwd, (dx, dy, x, weight, rstd, dwp, tm, T))

    dw = dwp.sum(0)
    if T != No:
        dw = dw[:No]

    return dx.view(*x_shape), dw.to(weight.dtype)


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
        offset=0.0,
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

        # Allocate rstd for backward (both paths now store it)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

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
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                rms_norm_kernel_static_persistent,
                (x_arg, y, weight, rstd, TILE_SIZE_M, TILE_SIZE_N, eps, offset),
            )
        else:
            # Standard mode
            if bias is not None:
                raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")

            MAX_FUSED_SIZE = 4096 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                rms_norm_kernel_gather,
                (
                    x_arg,
                    weight,
                    y,
                    rstd,
                    N,
                    eps,
                    offset,
                    TILE_SIZE,
                ),
            )

        # Always save for backward (both paths now produce rstd)
        ctx.save_for_backward(x, weight, rstd)
        ctx.TILE_SIZE = next_power_of_2(N)
        ctx.eps = eps
        ctx.offset = offset

        return y.view(*x.shape)

    @staticmethod
    def backward(ctx, dy):
        """
        Persistent backward pass using Bastile's grid-stride kernel.
        Supports backward from both gather and static persistent forward modes.
        """
        # Check if offset was used (backward not supported with non-zero offset)
        if ctx.offset != 0.0:
            raise NotImplementedError(
                f"Backward pass not implemented for CuTile RMSNorm with non-zero offset ({ctx.offset})"
            )

        x, weight, rstd = ctx.saved_tensors
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        M = x2.shape[0]
        dy2 = dy.reshape(-1, N)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()

        cfg = _bwd_cfg.get((M, N))
        if cfg is None:
            cfg = _bwd_tiles(M, N)
            _bwd_cfg[(M, N)] = cfg
        tm, T, g, No = cfg

        stream = torch.cuda.current_stream()

        dx = torch.empty_like(x2)
        dwp = torch.empty((g, T), device=x.device, dtype=torch.float32)
        ct.launch(stream, (g,), _rms_bwd, (dx, dy2, x2, weight, rstd, dwp, tm, T))

        dw = dwp.sum(0)
        if T != No:
            dw = dw[:No]

        # Gradients: (x, normalized_shape, weight, eps, bias, static_persistent, offset)
        return dx.view(shape), None, dw.to(weight.dtype), None, None, None, None


@register_impl("rms_norm", backend="cutile")
def rms_norm(input, normalized_shape, weight, eps, bias=None, static_persistent=None, offset=0.0, **kwargs):
    """
    Root mean square normalization implemented using CUDA Tile

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Normalization shape (for compatibility, not used)
        weight: Tensor of shape (N,)
        eps: Small constant added to variance calculation
        bias: Bias tensor of shape (N,), default is None (not supported in cutile)
        static_persistent: Whether to use static persistent kernel, default is False
        offset: Offset to add to weight (default 0.0 for Llama, 1.0 for Gemma3)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Normalized tensor with same shape as input
    """
    return RMSNorm.apply(input, normalized_shape, weight, eps, bias, static_persistent, offset)


class TileRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, offset=0.0):
        """
        RMSNorm implementation using CUDA Tile

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
            offset=self.offset,
        )

    def forward_torch(self, hidden_states):
        """PyTorch reference implementation for comparison"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.offset + self.weight) * hidden_states.to(input_dtype)

    @staticmethod
    def compute_rstd_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute rstd (reciprocal standard deviation) for RMSNorm using PyTorch. Simulates what the forward pass would save for backward."""
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
        """
        return rms_norm_backward(x, dy, weight, rstd)

    @staticmethod
    def rms_norm_backward_torch(
        x: torch.Tensor,
        dy: torch.Tensor,
        weight: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standalone RMSNorm backward pass using PyTorch. This is explicitly the torch reference implementation, not the cutile implementation."""
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        dy = dy.reshape(-1, dy.shape[-1])
        M, N = x.shape

        # Reshape rstd for broadcasting: (M,) -> (M, 1)
        rstd = rstd.view(M, 1)

        # Cast to fp32 up front so all intermediates are full precision
        x_f = x.float()
        dy_f = dy.float()
        w_f = weight.float()

        # Gradient w.r.t. weight: dw = sum((x * rstd) * dy, dim=0)
        x_norm = x_f * rstd
        dw = (dy_f * x_norm).sum(dim=0)

        # Gradient w.r.t. x
        dy_weighted = dy_f * w_f
        c1 = (dy_weighted * x_norm).sum(dim=1, keepdim=True)
        dx = rstd * (dy_weighted - x_norm * c1 / N)

        dx = dx.view(x_shape).to(x.dtype)
        dw = dw.to(weight.dtype)

        return dx, dw

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


class RMSNormForGemma3(TileRMSNorm):
    """
    RMSNorm implementation for Gemma3 models using CuTile backend.

    Gemma3 uses 'dim' parameter name instead of 'hidden_size', and initializes
    weights with zeros instead of ones, with offset=1.0.
    """

    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        # Initialize parent with offset
        super().__init__(hidden_size=dim, eps=eps, offset=offset)
        # Override weight initialization to zeros for Gemma3
        self.weight = nn.Parameter(torch.zeros(dim))

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"


@register_impl("get_rms_norm_module", backend="cutile")
def get_rms_norm_module(model: str = "llama"):
    if model == "gemma3":
        return RMSNormForGemma3
    else:
        return TileRMSNorm
