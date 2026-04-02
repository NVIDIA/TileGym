# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_sparse_mla(q, k, v, indices, qpe, kpe, is_causal=True, scaling=None, **kwargs):
    """PyTorch reference for sparse MLA."""
    qkv_dtype = v.dtype
    B, H, S, D = q.shape
    _, H_kv, S_kv, _ = k.shape
    _, _, _, topk = indices.shape
    D_PE = qpe.shape[-1]

    if scaling is None:
        scaling = 1.0 / math.sqrt(D + D_PE)

    q = q.float()
    k = k.float()
    v = v.float()
    qpe = qpe.float()
    kpe = kpe.float()

    if H != H_kv:
        assert H % H_kv == 0
        group_size = H // H_kv
        k = k.unsqueeze(2).expand(B, H_kv, group_size, S_kv, D).reshape(B, H, S_kv, D)
        v = v.unsqueeze(2).expand(B, H_kv, group_size, S_kv, D).reshape(B, H, S_kv, D)
        kpe_expanded = kpe.expand(B, H, S_kv, D_PE)
        indices = indices.unsqueeze(3).expand(B, S, H_kv, group_size, topk).reshape(B, S, H, topk)
    else:
        kpe_expanded = kpe.expand(B, H, S_kv, D_PE)

    idx = indices.long().permute(0, 2, 1, 3)

    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
    gathered_k = torch.gather(k, 2, idx_expanded.reshape(B, H, -1, D)).reshape(B, H, S, topk, D)
    gathered_v = torch.gather(v, 2, idx_expanded.reshape(B, H, -1, D)).reshape(B, H, S, topk, D)

    idx_pe = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D_PE)
    gathered_kpe = torch.gather(kpe_expanded, 2, idx_pe.reshape(B, H, -1, D_PE)).reshape(B, H, S, topk, D_PE)

    score = torch.einsum("bhsd,bhstd->bhst", q, gathered_k)
    score += torch.einsum("bhsd,bhstd->bhst", qpe, gathered_kpe)
    score *= scaling

    if is_causal:
        s_positions = torch.arange(S, device=q.device).view(1, 1, S, 1)
        causal_mask = idx <= s_positions
        score = score.masked_fill(~causal_mask, float("-inf"))

    attn = torch.softmax(score, dim=-1)
    out = torch.einsum("bhst,bhstd->bhsd", attn, gathered_v)
    return out.to(qkv_dtype)


register_impl("sparse_mla", "torch")(reference_sparse_mla)


ALL_BACKENDS = [
    ("cutile", "CuTile", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def _generate_indices(B, S, H_kv, topk, S_kv, device):
    """Unique past indices + future fillers for causal early tokens."""
    assert topk <= S_kv
    indices = torch.zeros(B, S, H_kv, topk, dtype=torch.int32, device=device)
    for b in range(B):
        for s in range(S):
            for g in range(H_kv):
                past_n = min(topk, s + 1)
                past = torch.randperm(s + 1, device=device)[:past_n].to(torch.int32)
                if past_n < topk:
                    future_n = topk - past_n
                    future = (torch.randperm(S_kv - s - 1, device=device)[:future_n] + s + 1).to(torch.int32)
                    idx = torch.cat([past, future])
                else:
                    idx = past
                indices[b, s, g, :] = idx[torch.randperm(topk, device=device)]
    return indices


def create_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None
    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["topk"],
        x_vals=[64, 128, 256, 512, 1024, 2048],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"sparse-mla-topk-scaling-{dtype_name}-TFLOPS",
        args={
            "dtype": dtype,
            "B": 1,
            "H": 16,
            "S": 256,
            "S_kv": 4096,
            "D": 128,
            "D_PE": 64,
            "H_kv": 1,
        },
    )


@triton.testing.perf_report([create_benchmark_config(dtype) for dtype in [torch.bfloat16]])
def bench_sparse_mla(topk, backend, dtype, B, H, S, S_kv, D, D_PE, H_kv, device=DEVICE):
    q = torch.empty(B, H, S, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    k = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    v = torch.empty(B, H_kv, S_kv, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    qpe = torch.empty(B, H, S, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    kpe = torch.empty(B, 1, S_kv, D_PE, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    indices = _generate_indices(B, S, H_kv, topk, S_kv, device)
    scaling = 1.0 / math.sqrt(D + D_PE)

    fn = lambda: tilegym.ops.sparse_mla(
        q,
        k,
        v,
        indices,
        qpe,
        kpe,
        is_causal=True,
        scaling=scaling,
        backend=backend,
    )
    if topk <= 256 and backend != "torch":
        ref = lambda: reference_sparse_mla(q, k, v, indices, qpe, kpe, scaling=scaling)
        torch.testing.assert_close(fn(), ref(), atol=1e-2, rtol=1e-2)

    ms = triton.testing.do_bench(fn)
    # QK(nope) + QK(rope) + PV = 2 * S * H * topk * (D + D_PE + D)
    total_flops = 2 * B * S * H * topk * (D + D_PE + D)
    return total_flops / (ms * 1e-3) / 1e12


if __name__ == "__main__":
    bench_sparse_mla.run(print_data=True)
