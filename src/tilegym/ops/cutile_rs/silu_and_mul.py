# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs silu_and_mul backend via FFI to libcutile_kernels.so (forward-only).

out = silu(input[..., :H]) * input[..., H:], where H = input.shape[-1] // 2.
Raw-pointer elementwise kernel: the input/output cross the cffi boundary as
``const TensorDesc*`` (the FFI reads ptr/shape/strides from them and wraps the
pointers as DevicePointer). BLOCK_SIZE = next_power_of_2(H) is a const generic.
"""

from types import SimpleNamespace

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import make_tensor_desc
from tilegym.backend.cutile_rs.utils import next_power_of_2

_KERNEL = "silu_and_mul"
_FFI_NAME = "cutile_silu_and_mul"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_silu_and_mul` signature in silu_and_mul_kernel/ffi.rs. hidden_size, row
# strides and the grid (n_rows) are derived inside the FFI from the descriptors.
_FFI_CDEF = """
int32_t cutile_silu_and_mul(
    const TensorDesc* out, const TensorDesc* inp,
    int32_t block_size, int32_t num_cta_in_cga, int32_t occupancy,
    int32_t device_id, uint64_t raw_stream);
"""

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_AUTO_COMPILE_OPTION = -1
_COMPILE_OCCUPANCY = None
_COMPILE_NUM_CTA_IN_CGA = None


def _compile_option_value(value) -> int:
    return _AUTO_COMPILE_OPTION if value is None else int(value)


def _configs():
    # BLOCK_SIZE is fixed by hidden_size (not tuned); a single auto config is the
    # faithful surface (occupancy / num_cta_in_cga left to the compiler default).
    # for sm100
    return [SimpleNamespace(OCCUPANCY=_COMPILE_OCCUPANCY, NUM_CTA_IN_CGA=_COMPILE_NUM_CTA_IN_CGA)]


def _run_ffi(out, x2, block_size, cfg):
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = out.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream
    od = make_tensor_desc(ffi, out)
    xd = make_tensor_desc(ffi, x2)
    rc = lib.cutile_silu_and_mul(
        od,
        xd,
        int(block_size),
        _compile_option_value(getattr(cfg, "NUM_CTA_IN_CGA", _COMPILE_NUM_CTA_IN_CGA)),
        _compile_option_value(getattr(cfg, "OCCUPANCY", _COMPILE_OCCUPANCY)),
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return out


def _forward_impl(x):
    if x.dtype not in _DTYPES:
        raise NotImplementedError(f"cutile-rs silu_and_mul: dtype {x.dtype} not supported")
    last = int(x.shape[-1])
    if last % 2 != 0:
        raise ValueError(f"cutile-rs silu_and_mul: input last dim must be even (2*hidden_size), got {last}")
    hidden_size = last // 2

    x_contig = x.contiguous()
    n_rows = int(x_contig.numel() // last)
    x2 = x_contig.reshape(n_rows, last)
    block_size = next_power_of_2(hidden_size)
    out_shape = tuple(x.shape[:-1]) + (hidden_size,)

    configs = _configs()

    def kernel_fn(cfg):
        out_local = torch.empty((n_rows, hidden_size), device=x.device, dtype=x.dtype)
        _run_ffi(out_local, x2, block_size, cfg)
        return out_local

    result = autotune_launch(
        kernel_fn=kernel_fn,
        configs=configs,
        key=(n_rows, hidden_size, x.dtype),
        kernel_name=_KERNEL,
    )
    return result.output.reshape(out_shape)


@register_impl("silu_and_mul", backend="cutile-rs")
def silu_and_mul(input, out=None, **kwargs):
    """SiLU-and-multiply via cutile-rs FFI (forward-only)."""
    if not input.is_cuda:
        raise ValueError("cutile-rs silu_and_mul: input must be a CUDA tensor")
    if input.requires_grad:
        input = input.detach()

    result = _forward_impl(input)

    if out is not None:
        if tuple(out.shape) != tuple(result.shape):
            raise ValueError(f"cutile-rs silu_and_mul: out shape {tuple(out.shape)} != {tuple(result.shape)}")
        out.copy_(result)
        return out
    return result
