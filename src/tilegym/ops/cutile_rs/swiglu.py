# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs swiglu backend via FFI to libcutile_kernels.so (forward-only).

c = SiLU(a) * b, with a/b the same shape. Raw-pointer elementwise kernel: a/b/c
cross the cffi boundary as ``const TensorDesc*`` (the FFI reads ptr/shape/strides
from them and wraps the pointers as DevicePointer). TILE_SIZE = next_power_of_2
(n_cols) is a const generic; grid = n_rows. Registered as ``get_swiglu`` (a getter
that returns the ``swiglu`` callable, matching the dispatch convention).
"""

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import make_tensor_desc
from tilegym.backend.cutile_rs.utils import next_power_of_2

_KERNEL = "swiglu"
_FFI_NAME = "cutile_swiglu"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_swiglu` signature in swiglu_kernel/ffi.rs. shapes/strides and the grid
# (n_rows) are derived inside the FFI from the descriptors.
_FFI_CDEF = """
int32_t cutile_swiglu(
    const TensorDesc* c, const TensorDesc* a, const TensorDesc* b,
    int32_t tile_size, int32_t num_cta_in_cga, int32_t occupancy,
    int32_t device_id, uint64_t raw_stream);
"""

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_AUTO_COMPILE_OPTION = -1


def _swiglu_forward(a, b):
    if a.dtype not in _DTYPES:
        raise NotImplementedError(f"cutile-rs swiglu: dtype {a.dtype} not supported")
    if b.dtype != a.dtype:
        raise NotImplementedError(f"cutile-rs swiglu requires a.dtype == b.dtype, got {a.dtype} / {b.dtype}")

    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a2 = a.contiguous().view(-1, n_cols)
    b2 = b.contiguous().view(-1, n_cols)
    c2 = torch.empty_like(a2)
    tile_size = next_power_of_2(int(n_cols))

    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = a2.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream
    cd = make_tensor_desc(ffi, c2)
    ad = make_tensor_desc(ffi, a2)
    bd = make_tensor_desc(ffi, b2)
    rc = lib.cutile_swiglu(
        cd,
        ad,
        bd,
        int(tile_size),
        _AUTO_COMPILE_OPTION,
        _AUTO_COMPILE_OPTION,
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return c2.view(*ori_shape)


def swiglu(a, b):
    """SwiGLU forward via cutile-rs FFI (forward-only): c = SiLU(a) * b."""
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("cutile-rs swiglu: a and b must be CUDA tensors on the same device")
    if a.requires_grad:
        a = a.detach()
    if b.requires_grad:
        b = b.detach()
    return _swiglu_forward(a, b)


@register_impl("get_swiglu", backend="cutile-rs")
def get_swiglu():
    return swiglu
