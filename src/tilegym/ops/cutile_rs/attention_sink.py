# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs attention_sink backend via FFI to libcutile_kernels.so (forward-only).

Flash-attention with attention-sink tokens. Inputs:
    * query : [bs, n_ctx, n_kv_heads, repeat_kv, head_dim]
    * key   : [bs, n_kv_ctx, n_kv_heads, head_dim]
    * value : [bs, n_kv_ctx, n_kv_heads, head_dim]
    * sinks : [n_heads]  (n_heads = n_kv_heads * repeat_kv)
    * output: [bs, n_ctx, n_heads * head_dim]

The kernel works on [bs, heads, seq, head_dim] contiguous tensors and launches a
2-D grid (ceil(n_ctx / TILE_M), bs * n_heads). q/k/v/out/sinks cross the cffi
boundary as ``const TensorDesc*`` (E dtype); start_q is an int32 ``TensorDesc``.
"""

import math
from types import SimpleNamespace

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import make_tensor_desc

_KERNEL = "attention_sink"
_FFI_NAME = "cutile_attention_sink"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_attention_sink` signature in attention_sink_kernel/ffi.rs.
_FFI_CDEF = """
int32_t cutile_attention_sink(
    const TensorDesc* out, const TensorDesc* q, const TensorDesc* k,
    const TensorDesc* v, const TensorDesc* sinks, const TensorDesc* start_q,
    float qk_scale,
    int32_t tile_d, int32_t h, int32_t n_kv_ctx,
    int32_t tile_m, int32_t tile_n, int32_t query_group_size, int32_t bandwidth,
    int32_t grid_x, int32_t grid_y,
    int32_t num_cta_in_cga, int32_t occupancy,
    int32_t device_id, uint64_t raw_stream);
"""

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_AUTO_COMPILE_OPTION = -1


def _configs():
    # attention autotune space (TILE_M, TILE_N, OCCUPANCY).
    # for sm100
    tile_ms, tile_ns, occs = (128, 64), (64,), (1, 2)
    return [SimpleNamespace(TILE_M=tm, TILE_N=tn, OCCUPANCY=occ) for tm in tile_ms for tn in tile_ns for occ in occs]


def _run_ffi(
    o,
    q,
    k,
    v,
    sinks,
    start_q_t,
    sm_scale,
    *,
    tile_m,
    tile_n,
    head_dim,
    n_heads,
    n_kv_ctx,
    repeat_kv,
    bandwidth,
    occupancy,
):
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = q.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream

    bs = int(q.shape[0])
    n_ctx = int(q.shape[2])
    grid_x = math.ceil(n_ctx / tile_m)
    grid_y = bs * n_heads
    occ = int(occupancy) if occupancy and occupancy > 0 else _AUTO_COMPILE_OPTION

    od = make_tensor_desc(ffi, o)
    qd = make_tensor_desc(ffi, q)
    kd = make_tensor_desc(ffi, k)
    vd = make_tensor_desc(ffi, v)
    sd = make_tensor_desc(ffi, sinks)
    stq = make_tensor_desc(ffi, start_q_t)
    rc = lib.cutile_attention_sink(
        od,
        qd,
        kd,
        vd,
        sd,
        stq,
        float(sm_scale),
        int(head_dim),
        int(n_heads),
        int(n_kv_ctx),
        int(tile_m),
        int(tile_n),
        int(repeat_kv),
        int(bandwidth),
        int(grid_x),
        int(grid_y),
        _AUTO_COMPILE_OPTION,
        occ,
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return o


@register_impl("attention_sink", backend="cutile-rs")
def attention_sink(
    query,
    key,
    value,
    sinks,
    sm_scale: float = 0.125,
    sliding_window=None,
    start_q=0,
    **kwargs,
):
    """Attention with sink tokens via cutile-rs FFI (forward-only)."""
    if query.dtype not in _DTYPES:
        raise NotImplementedError(f"cutile-rs attention_sink: dtype {query.dtype} not supported")

    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("cutile-rs attention_sink: query/key/value must be CUDA tensors")
    if not (query.device == key.device == value.device):
        raise ValueError("cutile-rs attention_sink: query/key/value must be on the same device")
    if query.requires_grad:
        query = query.detach()
    if key.requires_grad:
        key = key.detach()
    if value.requires_grad:
        value = value.detach()
    if isinstance(sinks, torch.Tensor) and sinks.requires_grad:
        sinks = sinks.detach()

    bs, n_ctx, n_kv_heads, repeat_kv, head_dim = query.shape
    _, n_kv_ctx, _, _ = key.shape
    n_heads = n_kv_heads * repeat_kv

    # Merge kv_heads/repeat_kv and move to [bs, heads, seq, head_dim] contiguous.
    q = query.view(bs, n_ctx, n_heads, head_dim).transpose(1, 2).contiguous()
    k = key.view(bs, n_kv_ctx, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    v = value.view(bs, n_kv_ctx, n_kv_heads, head_dim).transpose(1, 2).contiguous()
    sinks_c = sinks.contiguous()

    # start_q as an int32 GPU tensor (avoid .item() sync).
    if isinstance(start_q, torch.Tensor):
        start_q_t = start_q.to(torch.int32).contiguous()
        if start_q_t.device.type != "cuda":
            start_q_t = start_q_t.cuda()
    else:
        start_q_t = torch.tensor([int(start_q)], dtype=torch.int32, device=query.device)

    bandwidth = sliding_window if sliding_window is not None else 0

    def kernel_fn(cfg):
        o_local = torch.empty_like(q)
        _run_ffi(
            o_local,
            q,
            k,
            v,
            sinks_c,
            start_q_t,
            sm_scale,
            tile_m=int(cfg.TILE_M),
            tile_n=int(cfg.TILE_N),
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_ctx=n_kv_ctx,
            repeat_kv=repeat_kv,
            bandwidth=bandwidth,
            occupancy=getattr(cfg, "OCCUPANCY", _AUTO_COMPILE_OPTION),
        )
        return o_local

    result = autotune_launch(
        kernel_fn=kernel_fn,
        configs=_configs(),
        key=(bs, n_heads, n_ctx, head_dim, n_kv_ctx, bandwidth, query.dtype, str(query.device)),
        kernel_name=_KERNEL,
    )
    o = result.output
    # Back to [bs, n_ctx, heads, head_dim] -> merge heads.
    o = o.transpose(1, 2).contiguous()
    return o.view(bs, n_ctx, n_heads * head_dim)
