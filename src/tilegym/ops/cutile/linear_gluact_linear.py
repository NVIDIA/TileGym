# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile implementation of fused Linear + GLU activation + Linear."""

from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

# Module-level tune cache: (M, N1, K, act_type_id, dtype, device) -> (best_cfg, tuned_kernel)
_linear_gluact_tune_cache: dict = {}

# Activation type constants
ACT_SILU = 0
ACT_RELU = 1
ACT_GELU = 2

activation_type_map = {"silu": ACT_SILU, "relu": ACT_RELU, "gelu": ACT_GELU, "gelu-tanh": ACT_GELU}


def _sigmoid_ct(x, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    one = ct.full((BLOCK_M, BLOCK_N), 1.0, dtype=ct.float32)
    neg_x = -x
    exp_neg_x = ct.exp(neg_x)
    denom = one + exp_neg_x
    return ct.truediv(one, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)


def _silu_fwd_ct(x, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    sigmoid_x = _sigmoid_ct(x, BLOCK_M, BLOCK_N)
    return x * sigmoid_x


def _relu_fwd_ct(x, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    """ReLU activation: max(0, x)"""
    zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    return ct.maximum(x, zeros)


def _gelu_tanh_fwd_ct(x, BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int]):
    """GELU activation using tanh approximation"""
    sqrt_2_over_pi = ct.full((BLOCK_M, BLOCK_N), 0.7978845608028654, dtype=ct.float32)
    const_044715 = ct.full((BLOCK_M, BLOCK_N), 0.044715, dtype=ct.float32)
    half = ct.full((BLOCK_M, BLOCK_N), 0.5, dtype=ct.float32)
    one = ct.full((BLOCK_M, BLOCK_N), 1.0, dtype=ct.float32)

    x_squared = x * x
    x_cubed = x_squared * x
    inner_arg = x + const_044715 * x_cubed
    tanh_arg = sqrt_2_over_pi * inner_arg
    tanh_result = ct.tanh(tanh_arg)
    inner_term = half * (one + tanh_result)
    return x * inner_term


# AUTOTUNING CONFIGURATIONS
def _linear_gluact_autotune_configs():
    """
    Iterator of autotune configurations for linear_gluact kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] >= 10:
        # sm100+ (Blackwell): supports 2CTA and larger blocks
        # 2CTA (medium-large matrices)
        yield SimpleNamespace(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GROUP_M=8, num_ctas=2, occupancy=1)
        yield SimpleNamespace(BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_ctas=2, occupancy=1)
        yield SimpleNamespace(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=8, num_ctas=2, occupancy=1)
        # 1CTA (fallback + small matrices)
        yield SimpleNamespace(BLOCK_M=256, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_ctas=1, occupancy=1)
    else:
        # Older GPUs: only NUM_CTAS=1 is supported
        yield SimpleNamespace(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_ctas=1, occupancy=1)

    # Conservative fallback for small matrices and unsupported architectures.
    yield SimpleNamespace(BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, GROUP_M=8, num_ctas=1, occupancy=1)
    yield SimpleNamespace(BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_ctas=1, occupancy=1)


@ct.kernel
def _linear_gluact_fwd_kernel(
    Input,  # Input tensor [M, K]
    Weight_act,  # Weight for activation branch [N1, K]
    Weight_noact,  # Weight for non-activation branch [N1, K]
    Act_out,  # Output of activation branch [M, N1]
    Noact_out,  # Output of non-activation branch [M, N1]
    Mul_out,  # Output after GLU gating [M, N1]
    M: ct.Constant[int],
    N1: ct.Constant[int],
    K: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_M: ct.Constant[int],
    ACT_TYPE: ct.Constant[int],
):
    """
    cuTile kernel for fused Linear + GLU Activation forward pass.

    Computes:
        1. act_in = input @ weight_act^T
        2. act_out = activation(act_in)
        3. noact_out = input @ weight_noact^T
        4. mul_out = act_out * noact_out  [GLU gating]
    """
    pid = ct.bid(0)

    # Calculate grid dimensions
    grid_m = ct.cdiv(M, BLOCK_M)
    grid_n = ct.cdiv(N1, BLOCK_N)

    # Grid reordering for better cache locality
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = ct.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Initialize accumulators for both branches
    acc_act = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    acc_noact = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

    # Matrix multiplication loop over K dimension
    num_k_blocks = ct.cdiv(K, BLOCK_K)

    for k_block in range(num_k_blocks):
        # Load input tile [BLOCK_M, BLOCK_K] - keep in native dtype (fp16)
        input_tile = ct.load(Input, index=(pid_m, k_block), shape=(BLOCK_M, BLOCK_K))

        # Load weight_act tile [BLOCK_N, BLOCK_K] and transpose - keep in native dtype (fp16)
        weight_act_tile = ct.load(Weight_act, index=(pid_n, k_block), shape=(BLOCK_N, BLOCK_K))
        weight_act_tile_T = ct.transpose(weight_act_tile)

        # Load weight_noact tile [BLOCK_N, BLOCK_K] and transpose - keep in native dtype (fp16)
        weight_noact_tile = ct.load(Weight_noact, index=(pid_n, k_block), shape=(BLOCK_N, BLOCK_K))
        weight_noact_tile_T = ct.transpose(weight_noact_tile)

        # Accumulate: input @ weight^T (tensor cores handle fp16 input -> fp32 accumulation)
        acc_act = ct.mma(input_tile, weight_act_tile_T, acc=acc_act)
        acc_noact = ct.mma(input_tile, weight_noact_tile_T, acc=acc_noact)

    # Convert to output dtype before activation (fp16 activation is as accurate as fp32)
    acc_act = ct.astype(acc_act, Input.dtype)
    acc_noact = ct.astype(acc_noact, Input.dtype)

    # Apply activation (activation functions compute in fp32 internally for precision)
    if ACT_TYPE == ACT_SILU:
        act_out_tile = _silu_fwd_ct(acc_act, BLOCK_M, BLOCK_N)
    elif ACT_TYPE == ACT_RELU:
        act_out_tile = _relu_fwd_ct(acc_act, BLOCK_M, BLOCK_N)
    elif ACT_TYPE == ACT_GELU:
        act_out_tile = _gelu_tanh_fwd_ct(acc_act, BLOCK_M, BLOCK_N)
    else:
        act_out_tile = acc_act

    # Convert activation output back to input dtype
    act_out_tile = ct.astype(act_out_tile, Input.dtype)

    mul_out_tile = act_out_tile * acc_noact

    # Store all outputs
    ct.store(Act_out, index=(pid_m, pid_n), tile=act_out_tile)
    ct.store(Noact_out, index=(pid_m, pid_n), tile=acc_noact)
    ct.store(Mul_out, index=(pid_m, pid_n), tile=mul_out_tile)


def _cutile_autotune_linear_gluact(
    stream,
    input_flat,
    weight_act,
    weight_noact,
    act_out,
    noact_out,
    mul_out,
    M,
    N1,
    K,
    act_type_id,
):
    """
    Autotuned launch for linear_gluact_fwd_kernel.
    """
    cache_key = (M, N1, K, act_type_id, input_flat.dtype, str(input_flat.device))
    if cache_key not in _linear_gluact_tune_cache:
        result = exhaustive_search(
            list(_linear_gluact_autotune_configs()),
            stream,
            lambda cfg: (ceil(M / cfg.BLOCK_M) * ceil(N1 / cfg.BLOCK_N), 1, 1),
            _linear_gluact_fwd_kernel,
            lambda cfg: (
                input_flat,
                weight_act,
                weight_noact,
                act_out,
                noact_out,
                mul_out,
                M,
                N1,
                K,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_M,
                act_type_id,
            ),
            lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _linear_gluact_tune_cache[cache_key] = (
            best_cfg,
            _linear_gluact_fwd_kernel.replace_hints(num_ctas=best_cfg.num_ctas, occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _linear_gluact_tune_cache[cache_key]
    ct.launch(
        stream,
        (ceil(M / best_cfg.BLOCK_M) * ceil(N1 / best_cfg.BLOCK_N), 1, 1),
        tuned_kernel,
        (
            input_flat,
            weight_act,
            weight_noact,
            act_out,
            noact_out,
            mul_out,
            M,
            N1,
            K,
            best_cfg.BLOCK_M,
            best_cfg.BLOCK_N,
            best_cfg.BLOCK_K,
            best_cfg.GROUP_M,
            act_type_id,
        ),
    )


def _linear_gluact_linear_cutile_impl(
    input: torch.Tensor,
    weight_act: torch.Tensor,
    weight_noact: torch.Tensor,
    weight2: torch.Tensor,
    act_type: str = "silu",
    kernel_configs: dict = None,
    use_autotune: bool = True,
):
    """
    cuTile implementation of Linear + GLU Activation + Linear.

    Computation Flow (GLU - Gated Linear Unit):
        1. input -> Linear1_act   : act_in = input @ weight_act^T
        2. act_in -> activation   : act_out = activation(act_in)
        3. input -> Linear1_noact : noact_out = input @ weight_noact^T
        4. element-wise multiply  : mul_out = act_out * noact_out  [gating]
        5. mul_out -> Linear2     : output = mul_out @ weight2^T

    Args:
        input: Input tensor (*, in_features)
        weight_act: Weight for activation branch (out_features, in_features)
        weight_noact: Weight for non-activation branch (out_features, in_features)
        weight2: Weight for final linear (final_features, out_features)
        act_type: Activation type ('silu', 'relu', 'gelu', 'gelu-tanh')
        kernel_configs: Kernel configuration dict (ignored if use_autotune=True)
        use_autotune: Whether to use autotuning (default: True)

    Returns:
        output: Output tensor (*, final_features)
    """
    # Validate activation type
    assert act_type in activation_type_map, f"Unsupported activation type: {act_type}"
    act_type_id = activation_type_map[act_type]

    # Get dimensions
    input_shape = input.shape
    if input.dim() > 2:
        input_flat = input.view(-1, input_shape[-1])
    else:
        input_flat = input

    M, K = input_flat.shape
    N1, K_weight = weight_act.shape
    N2, N1_weight2 = weight2.shape

    assert K == K_weight, f"Input/weight dimension mismatch: {K} != {K_weight}"
    assert N1 == N1_weight2, f"Weight dimension mismatch: {N1} != {N1_weight2}"
    assert weight_act.shape == weight_noact.shape, "weight_act and weight_noact must have same shape"

    # Ensure contiguous
    assert input_flat.is_contiguous(), "Input must be contiguous"
    assert weight_act.is_contiguous(), "weight_act must be contiguous"
    assert weight_noact.is_contiguous(), "weight_noact must be contiguous"
    assert weight2.is_contiguous(), "weight2 must be contiguous"

    # Allocate intermediate tensors
    act_out = torch.empty((M, N1), device=input.device, dtype=input.dtype)
    noact_out = torch.empty((M, N1), device=input.device, dtype=input.dtype)
    mul_out = torch.empty((M, N1), device=input.device, dtype=input.dtype)

    stream = torch.cuda.current_stream()

    if use_autotune:
        # Launch autotuned kernel
        _cutile_autotune_linear_gluact(
            stream,
            input_flat,
            weight_act,
            weight_noact,
            act_out,
            noact_out,
            mul_out,
            M,
            N1,
            K,
            act_type_id,
        )
    else:
        # Use manual kernel configuration
        default_configs = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
        }
        if kernel_configs is not None:
            default_configs.update(kernel_configs)

        BLOCK_M = default_configs["BLOCK_M"]
        BLOCK_N = default_configs["BLOCK_N"]
        BLOCK_K = default_configs["BLOCK_K"]
        GROUP_M = default_configs["GROUP_M"]

        grid = (ceil(M / BLOCK_M) * ceil(N1 / BLOCK_N), 1, 1)

        ct.launch(
            stream,
            grid,
            _linear_gluact_fwd_kernel,
            (
                input_flat,
                weight_act,
                weight_noact,
                act_out,
                noact_out,
                mul_out,
                M,
                N1,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_M,
                act_type_id,
            ),
        )

    # Final linear transformation using cuTile matmul
    from tilegym.ops.cutile.matmul import matmul as cutile_matmul

    output = cutile_matmul(mul_out, weight2, trans_b=True, static_persistent=True)

    # Reshape output if needed
    if input.dim() > 2:
        output = output.view(*input_shape[:-1], N2)

    return output


@register_impl("linear_gluact_linear", backend="cutile")
def linear_gluact_linear(
    input: torch.Tensor,
    weight_act: torch.Tensor,
    weight_noact: torch.Tensor,
    weight2: torch.Tensor,
    act_type: str = "silu",
    kernel_configs: dict = None,
):
    """
    Registered cuTile implementation for linear_gluact_linear dispatch.

    Args:
        input: Input tensor (*, in_features)
        weight_act: Weight for activation branch (out_features, in_features)
        weight_noact: Weight for non-activation branch (out_features, in_features)
        weight2: Weight for final linear (final_features, out_features)
        act_type: Activation type ('silu', 'relu', 'gelu', 'gelu-tanh')
        kernel_configs: Kernel configuration dict (ignored if use_autotune=True)
    """
    return _linear_gluact_linear_cutile_impl(input, weight_act, weight_noact, weight2, act_type, kernel_configs)
