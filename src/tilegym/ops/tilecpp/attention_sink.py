# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.autotuner import Config
from tilegym.ops.tilecpp.autotuner import SearchSpace
from tilegym.ops.tilecpp.autotuner import TileCppAutotuner
from tilegym.ops.tilecpp.autotuner import is_autotuning_enabled
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_cpp_type

_attn_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention_sink.cuh", kernel_name="attention_sink_fwd_kernel"
)
_tuners = {}


def _configs(cap):
    for m in [128, 64] if cap[0] < 9 else [256, 128, 64]:
        for n in [64] if cap[0] < 9 else [128, 64]:
            for occ in [1, 2] if cap[0] < 9 else [1, 2, 4]:
                yield Config(TILE_M=m, TILE_N=n, num_ctas=1, occupancy=occ)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sinks, sm_scale, bandwidth, start_q):
        assert start_q.numel() == 1, "start_q must contain exactly one element"
        bs, nq, kv_heads, group, dim = query.shape
        nk = key.shape[1]
        heads = kv_heads * group
        q = query.view(bs, nq, heads, dim).transpose(1, 2).contiguous()
        k = key.view(bs, nk, kv_heads, dim).transpose(1, 2).contiguous()
        v = value.view(bs, nk, kv_heads, dim).transpose(1, 2).contiguous()
        out = torch.empty_like(q)
        bandwidth = 0 if bandwidth is None else bandwidth
        sink_type = get_cpp_type(sinks.dtype if sinks is not None else q.dtype)
        aligned = all(t.data_ptr() % 16 == 0 for t in (q, k, v, out))

        def launch(cfg):
            kernel, _, _ = _attn_fwd_kernel.get_kernel(
                dtype=q.dtype,
                template_params=[
                    sink_type,
                    dim,
                    heads,
                    nk,
                    group,
                    bandwidth,
                    cfg.TILE_M,
                    cfg.TILE_N,
                    cfg.num_ctas,
                    cfg.occupancy,
                    aligned,
                ],
                signature=f"const {{T}}*, const {{T}}*, const {{T}}*, const {sink_type}*, {{T}}*, const int*, float, int, int",
            )
            _attn_fwd_kernel.launch(
                grid=((nq + cfg.TILE_M - 1) // cfg.TILE_M, bs * heads, 1),
                kernel=kernel,
                args=[
                    np.uint64(q.data_ptr()),
                    np.uint64(k.data_ptr()),
                    np.uint64(v.data_ptr()),
                    np.uint64(sinks.data_ptr() if sinks is not None else 0),
                    np.uint64(out.data_ptr()),
                    np.uint64(start_q.data_ptr()),
                    np.float32(sm_scale),
                    np.int32(bs),
                    np.int32(nq),
                ],
            )

        if is_autotuning_enabled():
            cap = torch.cuda.get_device_capability(q.device)
            if cap not in _tuners:
                _tuners[cap] = TileCppAutotuner(SearchSpace(list(_configs(cap))))
            _tuners[cap](
                torch.cuda.current_stream(),
                key=(bs, heads, nq, dim, nk, bandwidth, q.dtype, str(q.device)),
                launch_fn=launch,
                grid_fn=lambda args, cfg: ((nq + cfg.TILE_M - 1) // cfg.TILE_M, bs * heads, 1),
            )
        else:
            launch(Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2))
        return out.transpose(1, 2).contiguous().view(bs, nq, heads * dim)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass for tilecpp attention_sink is not implemented.")


attention = _Attention.apply


@register_impl("attention_sink", backend="tilecpp")
def attention_sink(query, key, value, sinks, sm_scale=0.125, sliding_window=None, start_q=0, **kwargs):
    if isinstance(start_q, torch.Tensor):
        start_q_tensor = start_q.to(device=query.device, dtype=torch.int32).contiguous()
    else:
        start_q_tensor = torch.tensor([int(start_q)], dtype=torch.int32, device=query.device)
    return attention(query, key, value, sinks, sm_scale, sliding_window, start_q_tensor)
