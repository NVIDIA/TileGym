# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn.functional as F
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_swa(q, k, v, window_size, is_causal=True, scaling=None, **kwargs):
    """PyTorch reference: full materialized mask, O(S^2)."""
    B, H, S_Q, D = q.shape
    S_K = k.shape[2]
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scaling
    i = torch.arange(S_Q, device=q.device).unsqueeze(1)
    j = torch.arange(S_K, device=q.device).unsqueeze(0)
    mask = j > (i - window_size)
    if is_causal:
        mask = mask & (j <= i)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


register_impl("swa_attention", "torch")(reference_swa)


ALL_BACKENDS = [
    ("cutile", "CuTile SWA", ("blue", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch ref", ("green", "-")),
]


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None
    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=f"swa-attention-seq-scaling-{dtype_name}-TFLOPS",
        args={
            "dtype": dtype,
            "B": 1,
            "H": 32,
            "D": 128,
            "W": 4096,
        },
    )


@triton.testing.perf_report([create_benchmark_config(dtype) for dtype in [torch.float16]])
def bench_swa_attention(seq_len, backend, dtype, B, H, D, W, device=DEVICE):
    q = torch.empty(B, H, seq_len, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    k = torch.empty(B, H, seq_len, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)
    v = torch.empty(B, H, seq_len, D, device=device, dtype=dtype).normal_(mean=0.0, std=0.3)

    eff_w = min(W, seq_len)

    fn = lambda: tilegym.ops.swa_attention(
        q,
        k,
        v,
        window_size=eff_w,
        is_causal=True,
        backend=backend,
    )

    # spot-check correctness at small sizes
    if seq_len <= 2048 and backend != "torch":
        ref = lambda: reference_swa(q, k, v, window_size=eff_w, is_causal=True)
        torch.testing.assert_close(fn(), ref(), atol=5e-2, rtol=1e-2)

    ms = triton.testing.do_bench(fn)
    # 2 matmuls per KV block: QK^T and PV, each 2*M*N*K FLOPs
    total_flops = 2 * B * H * seq_len * eff_w * D * 2
    return total_flops / (ms * 1e-3) / 1e12


if __name__ == "__main__":
    bench_swa_attention.run(print_data=True)
