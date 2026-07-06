# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs bmm (batched matmul) backend via FFI to libcutile_kernels.so (forward-only).

C[Q, M, N] = A[Q, M, K] @ B[Q, K, N], optionally transposing A and/or B. Two
structural variants, selected by ``static_persistent``:
    * static_persistent=True  -> static_persistent_bmm_kernel
        generics <E, BM, BN, BK, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B>,
        persistent 1-D grid-stride launch, transpose supported, num_ctas tuned.
    * static_persistent=False -> non_persistent_bmm_kernel
        generics <E, BM, BN, BK>, 3-D direct launch, NO transpose.

``use_tma`` is ignored (the kernel always uses TMA view loads). Tensors cross the
cffi boundary as ``const TensorDesc*`` (dtype/shapes/strides travel inside).
"""

from types import SimpleNamespace

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import get_num_sm
from tilegym.backend.cutile_rs.utils import make_tensor_desc

_KERNEL = "bmm"
_FFI_NAME = "cutile_bmm"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_bmm` signature in bmm_kernel/ffi.rs. Tensors cross as `const TensorDesc*`
# (typedef prepended by bind_kernel_function_cffi).
_FFI_CDEF = """
int32_t cutile_bmm(
    const TensorDesc* c, const TensorDesc* a, const TensorDesc* b,
    int32_t bm, int32_t bn, int32_t bk,
    int32_t group_size_m, int32_t trans_a, int32_t trans_b,
    int32_t persistent, int32_t num_cta_in_cga, int32_t occupancy,
    int32_t num_programs, int32_t device_id, uint64_t raw_stream);
"""

# Supported dtypes (wrapper input validation; the dtype code is packed by
# make_tensor_desc into TensorDesc.dtype).
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_AUTO_COMPILE_OPTION = -1


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _persistent_configs():
    # static_persistent autotune configs (num_ctas=2 carried to FFI).
    # for sm100
    return [
        SimpleNamespace(BM=256, BN=256, BK=64, GROUP_SIZE_M=8, NUM_CTA_IN_CGA=2, OCCUPANCY=1),
        SimpleNamespace(BM=128, BN=256, BK=64, GROUP_SIZE_M=8, NUM_CTA_IN_CGA=2, OCCUPANCY=1),
    ]


def _non_persistent_configs():
    # non_persistent: fixed TILE 128/128/32, compile options auto.
    # for sm100
    return [
        SimpleNamespace(BM=128, BN=128, BK=32, GROUP_SIZE_M=8, NUM_CTA_IN_CGA=None, OCCUPANCY=None),
    ]


def _compile_option_value(value) -> int:
    return _AUTO_COMPILE_OPTION if value is None else int(value)


def _persistent_grid(total_tiles: int, num_ctas: int, occupancy: int) -> int:
    num_sms = get_num_sm()
    return max(min(num_sms // max(num_ctas, 1), total_tiles), 1) * max(occupancy, 1)


def _run_ffi(c, a, b, cfg, persistent, trans_a, trans_b):
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = a.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream

    bm, bn, bk = int(cfg.BM), int(cfg.BN), int(cfg.BK)
    group_size_m = int(getattr(cfg, "GROUP_SIZE_M", 8))
    num_cta_in_cga = getattr(cfg, "NUM_CTA_IN_CGA", None)
    occupancy = getattr(cfg, "OCCUPANCY", None)

    # logical (post-transpose) dims for the persistent grid math
    q = int(a.shape[0])
    m = int(a.shape[2]) if trans_a else int(a.shape[1])
    n = int(b.shape[1]) if trans_b else int(b.shape[2])
    if persistent:
        total_tiles = _cdiv(m, bm) * _cdiv(n, bn) * q
        num_programs = _persistent_grid(total_tiles, num_cta_in_cga or 1, occupancy or 1)
    else:
        # 3-D grid is computed inside the FFI; num_programs unused for this variant.
        num_programs = 1

    cd = make_tensor_desc(ffi, c)
    ad = make_tensor_desc(ffi, a)
    bd = make_tensor_desc(ffi, b)
    rc = lib.cutile_bmm(
        cd,
        ad,
        bd,
        bm,
        bn,
        bk,
        group_size_m,
        int(1 if trans_a else 0),
        int(1 if trans_b else 0),
        int(1 if persistent else 0),
        _compile_option_value(num_cta_in_cga),
        _compile_option_value(occupancy),
        int(num_programs),
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return c


@register_impl("bmm", backend="cutile-rs")
def bmm(a, b, transpose_a=False, transpose_b=False, static_persistent=True, use_tma=None, **kwargs):
    """batched matmul via cutile-rs FFI (forward-only). C = A @ B per batch."""
    if a.dtype not in _DTYPES:
        raise NotImplementedError(f"cutile-rs bmm: dtype {a.dtype} not supported")
    if b.dtype != a.dtype:
        raise NotImplementedError("cutile-rs bmm: a and b must share dtype")
    if a.ndim != 3 or b.ndim != 3:
        raise NotImplementedError("cutile-rs bmm: inputs must be rank-3 [Q, *, *]")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("cutile-rs bmm: a and b must be CUDA tensors on the same device")

    persistent = bool(static_persistent)
    if not persistent and (transpose_a or transpose_b):
        raise NotImplementedError("cutile-rs bmm non-persistent variant does not support transpose")

    if a.requires_grad:
        a = a.detach()
    if b.requires_grad:
        b = b.detach()

    a = a.contiguous()
    b = b.contiguous()

    q = int(a.shape[0])
    m = int(a.shape[2]) if transpose_a else int(a.shape[1])
    n = int(b.shape[1]) if transpose_b else int(b.shape[2])
    k = int(a.shape[1]) if transpose_a else int(a.shape[2])
    out_dtype = a.dtype

    configs = _persistent_configs() if persistent else _non_persistent_configs()

    def launch_with_cfg(cfg):
        out = torch.empty((q, m, n), device=a.device, dtype=out_dtype)
        _run_ffi(out, a, b, cfg, persistent, transpose_a, transpose_b)
        return out

    def kernel_fn(cfg):
        return launch_with_cfg(cfg)

    result = autotune_launch(
        kernel_fn=kernel_fn,
        configs=configs,
        key=(q, m, n, k, a.dtype, persistent, transpose_a, transpose_b),
        kernel_name=_KERNEL,
    )
    return result.output
