# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Persistent Layer Normalization.
"""

from math import ceil
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.autotuner import Config
from tilegym.ops.tilecpp.autotuner import SearchSpace
from tilegym.ops.tilecpp.autotuner import TileCppAutotuner
from tilegym.ops.tilecpp.autotuner import autotune
from tilegym.ops.tilecpp.autotuner import is_autotuning_enabled
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "persistent_layer_norm.cuh",
    kernel_name="persistent_layer_norm_fwd_kernel",
)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _persistent_layer_norm_autotune_configs():
    return [
        Config(BLOCK_N=2, num_ctas=1, occupancy=1),
        Config(BLOCK_N=4, num_ctas=1, occupancy=1),
        Config(BLOCK_N=8, num_ctas=1, occupancy=1),
        Config(BLOCK_N=16, num_ctas=1, occupancy=1),
        Config(BLOCK_N=32, num_ctas=1, occupancy=1),
    ]


def _register_pressure_predicate(named_args: dict, cfg: Config) -> bool:
    block_n = cfg.BLOCK_N
    block_d = named_args["BLOCK_D"]
    return (block_n * block_d) / (8 * 32) <= 256


def _get_default_persistent_layer_norm_configs():
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability[0] < 9:
        return {"BLOCK_N": 4, "num_ctas": 1, "occupancy": 1}
    return {"BLOCK_N": 8, "num_ctas": 1, "occupancy": 1}


def _launch_fwd(x, y, weight, bias, mean, rstd, eps, compute_mean_and_rstd, block_n, block_d):
    dtype = x.dtype
    dump_kernel_types("persistent_layer_norm_fwd_kernel", x, weight, bias)

    N, D = x.shape
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    num_row_blocks = (N + block_n - 1) // block_n
    grid_size = min(NUM_SMS, num_row_blocks)

    bool_to_str = lambda b: "true" if b else "false"
    IS_SWISH = False
    TRAINING = True

    # Format eps as a C++ float literal so it round-trips cleanly through the
    # template signature mangling.
    eps_literal = f"{float(eps):.17g}f"
    template_params = [
        block_n,
        block_d,
        bool_to_str(IS_SWISH),
        bool_to_str(TRAINING),
        bool_to_str(compute_mean_and_rstd),
        N,
        D,
        grid_size,
        eps_literal,
    ]

    kernel, _, _ = _fwd_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature=("const {T}*, {T}*, const {T}*, const {T}*, float*, float*"),
    )

    _fwd_kernel.launch(
        grid=(grid_size, 1, 1),
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(y.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(bias.data_ptr()),
            np.uint64(mean.data_ptr()),
            np.uint64(rstd.data_ptr()),
        ],
    )


# Reuse one SearchSpace instance so predicate_fn is attached.
_AUTOTUNE_SEARCH_SPACE = SearchSpace(
    _persistent_layer_norm_autotune_configs(),
    predicate_fn=_register_pressure_predicate,
)


@autotune(search_space=_AUTOTUNE_SEARCH_SPACE)
def _tilecpp_autotune_persistent_layer_norm(
    x,
    y,
    weight,
    bias,
    mean,
    rstd,
    eps,
    compute_mean_and_rstd,
    block_d,
    autotuner: TileCppAutotuner | None = None,
):
    """Autotuned launch — picks fastest BLOCK_N for (dtype, N, D) at runtime."""
    N, D = x.shape
    key = (
        "tilecpp_persistent_layer_norm",
        str(x.dtype),
        N,
        D,
        bool(compute_mean_and_rstd),
    )
    named_args = {"N": N, "D": D, "BLOCK_D": block_d}

    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count

    def launch_fn(cfg: Config):
        _launch_fwd(
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            eps,
            compute_mean_and_rstd,
            block_n=cfg.BLOCK_N,
            block_d=block_d,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple:
        num_row_blocks = ceil(args["N"] / cfg.BLOCK_N)
        return (min(NUM_SMS, num_row_blocks), 1, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args=named_args,
    )


def _persistent_layer_norm_fwd(x, weight, bias, eps, mean=None, rstd=None):
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = _switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1 and bias.dim() == 1
    assert weight.numel() == D and bias.numel() == D

    y = torch.empty_like(x)
    if (mean is None) != (rstd is None):
        raise ValueError(
            "persistent_layer_norm requires both mean and rstd to be supplied or both to be None; "
            "passing only one would silently overwrite it."
        )
    compute_mean_and_rstd = mean is None
    if compute_mean_and_rstd:
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

    BLOCK_D = _next_power_of_2(D)
    if BLOCK_D != D:
        raise ValueError(
            f"persistent_layer_norm requires the last dimension D to be a power of two "
            f"(got D={D}); the kernel does unmasked tile loads of size BLOCK_D=next_pow2(D)."
        )

    gpu_capability = torch.cuda.get_device_capability(x.device)
    enable_autotune = is_autotuning_enabled() and gpu_capability[0] >= 9

    if enable_autotune:
        _tilecpp_autotune_persistent_layer_norm(
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            eps,
            compute_mean_and_rstd,
            BLOCK_D,
        )
    else:
        configs = _get_default_persistent_layer_norm_configs()
        _launch_fwd(
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            eps,
            compute_mean_and_rstd,
            configs["BLOCK_N"],
            BLOCK_D,
        )

    num_warps = 8
    return y, mean, rstd, BLOCK_D, num_warps


@register_impl("persistent_layer_norm", backend="tilecpp")
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
    original_shape = input.shape
    if input.dim() != 2:
        input = input.reshape(-1, input.shape[-1])

    y, mean_out, rstd_out, block_d, num_warps = _persistent_layer_norm_fwd(
        input,
        weight,
        bias,
        eps,
        mean,
        rstd,
    )

    if len(original_shape) != 2:
        y = y.reshape(original_shape)

    return y, mean_out, rstd_out, block_d, num_warps
