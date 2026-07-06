# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs matmul backend via FFI to libcutile_kernels.so (forward-only).

C = A @ B. Two structural variants, selected by ``static_persistent``:
    * static_persistent=False -> non_persistent_matmul_kernel
        generics <E, BM, BN, BK, CAST_TF32>, flat 1-D grid cdiv(M,BM)*cdiv(N,BN).
    * static_persistent=True  -> static_persistent_matmul_kernel
        generics <E, BM, BN, BK, GROUP_SIZE_M, CAST_TF32>, runtime m,n,k,
        persistent grid-stride loop.

``use_tma`` is ignored by the wrapper (the kernel always uses TMA view loads);
``static_persistent`` selects the variant. trans_a / trans_b are not supported.
The reference autotunes num_ctas {1,2,4} -> forwarded to FFI num_cta_in_cga so
the CGA cluster size actually takes effect.
"""

from types import SimpleNamespace

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import get_num_sm
from tilegym.backend.cutile_rs.utils import make_tensor_desc

_KERNEL = "matmul"
_FFI_NAME = "cutile_matmul"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_matmul` signature in matmul_kernel/ffi.rs. Tensors cross as
# `const TensorDesc*` (typedef prepended by bind_kernel_function_cffi); dtype,
# shapes (m/n/k) and strides are carried in the descriptors.
_FFI_CDEF = """
int32_t cutile_matmul(
    const TensorDesc* c, const TensorDesc* a, const TensorDesc* b,
    int32_t bm, int32_t bn, int32_t bk,
    int32_t group_size_m, int32_t num_programs,
    int32_t num_cta_in_cga, int32_t occupancy,
    int32_t persistent, int32_t device_id, uint64_t raw_stream);
"""

# Supported dtypes (the code carried in TensorDesc.dtype is computed by
# make_tensor_desc; this set is only for the wrapper's input validation).
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_GROUP_SIZE_M = 8


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _non_persistent_configs():
    # non_persistent autotune configs.
    # for sm100
    return [
        SimpleNamespace(BM=128, BN=128, BK=32, NUM_CTAS=1, OCCUPANCY=1),
        SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTAS=2, OCCUPANCY=1),
        SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTAS=4, OCCUPANCY=1),
        SimpleNamespace(BM=512, BN=256, BK=64, NUM_CTAS=2, OCCUPANCY=1),
    ]


def _static_persistent_configs():
    # static_persistent autotune configs.
    # for sm100
    return [
        SimpleNamespace(BM=128, BN=512, BK=64, NUM_CTAS=4, OCCUPANCY=1),
        SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTAS=2, OCCUPANCY=1),
        SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTAS=1, OCCUPANCY=1),
        SimpleNamespace(BM=256, BN=256, BK=128, NUM_CTAS=2, OCCUPANCY=1),
    ]


def _non_persistent_grid(m: int, n: int, bm: int, bn: int) -> int:
    return _cdiv(m, bm) * _cdiv(n, bn)


def _static_persistent_grid(m: int, n: int, bm: int, bn: int, num_ctas: int, occupancy: int) -> int:
    num_tiles = _cdiv(m, bm) * _cdiv(n, bn)
    num_sms = get_num_sm()
    return max(min(num_sms // max(num_ctas, 1), num_tiles), 1) * max(occupancy, 1)


def _run_ffi(out, a, b, bm, bn, bk, num_programs, persistent, num_cta_in_cga, occupancy):
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = a.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream
    # Pack tensors via the shared descriptor (dtype/shape/strides travel inside).
    cd = make_tensor_desc(ffi, out)
    ad = make_tensor_desc(ffi, a)
    bd = make_tensor_desc(ffi, b)
    rc = lib.cutile_matmul(
        cd,
        ad,
        bd,
        int(bm),
        int(bn),
        int(bk),
        int(_GROUP_SIZE_M),
        int(num_programs),
        int(num_cta_in_cga),
        int(occupancy),
        int(persistent),
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return out


@register_impl("matmul", backend="cutile-rs")
def matmul(a, b, trans_a=None, trans_b=None, static_persistent=None, use_tma=None, **kwargs):
    """matmul via cutile-rs FFI (forward-only). C = A @ B."""
    if trans_a:
        raise NotImplementedError("cutile-rs matmul: trans_a is not supported")
    if trans_b:
        raise NotImplementedError("cutile-rs matmul: trans_b is not supported")

    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("cutile-rs matmul: a and b must be CUDA tensors on the same device")
    if a.requires_grad:
        a = a.detach()
    if b.requires_grad:
        b = b.detach()

    a = a.contiguous()
    b = b.contiguous()

    if a.dtype not in _DTYPES:
        raise NotImplementedError(f"cutile-rs matmul: dtype {a.dtype} not supported")
    if b.dtype != a.dtype:
        raise NotImplementedError("cutile-rs matmul: a and b must share dtype")

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"cutile-rs matmul: K mismatch: {k} vs {k2}")

    persistent = 1 if static_persistent else 0
    out_dtype = a.dtype

    if persistent:
        configs = _static_persistent_configs()
    else:
        configs = _non_persistent_configs()

    def launch_with_cfg(cfg):
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
        if persistent:
            num_programs = _static_persistent_grid(m, n, cfg.BM, cfg.BN, cfg.NUM_CTAS, cfg.OCCUPANCY)
        else:
            num_programs = _non_persistent_grid(m, n, cfg.BM, cfg.BN)
        _run_ffi(
            out,
            a,
            b,
            cfg.BM,
            cfg.BN,
            cfg.BK,
            num_programs,
            persistent,
            cfg.NUM_CTAS,
            cfg.OCCUPANCY,
        )
        return out

    def kernel_fn(cfg):
        return launch_with_cfg(cfg)

    result = autotune_launch(
        kernel_fn=kernel_fn,
        configs=configs,
        key=(m, n, k, a.dtype, persistent),
        kernel_name=_KERNEL,
    )
    return result.output
