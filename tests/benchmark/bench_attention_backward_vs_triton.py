#!/usr/bin/env python3

"""
Performance benchmarks for Flash Attention: CuTile vs Triton Flash Attention.

This file depends on an external triton-fused-attention.py and is NOT committed
to the TileGym repo. It is for local comparison only.

Usage:
    TRITON_FUSED_ATTN_PATH=/path/to/triton-fused-attention.py \
        python tests/benchmark/bench_attention_backward_vs_triton.py
"""

import math
import os
import sys

import torch
import triton
import triton.testing

from tilegym.backend import is_backend_available
from tilegym.ops.cutile.attention import tile_fmha_with_backward

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Load triton flash attention from external file
TRITON_FUSED_ATTN_PATH = os.environ.get(
    "TRITON_FUSED_ATTN_PATH", "/root/triton-fused-attention.py"
)
if not os.path.exists(TRITON_FUSED_ATTN_PATH):
    print(f"Error: {TRITON_FUSED_ATTN_PATH} not found.")
    print("Set TRITON_FUSED_ATTN_PATH to the path of triton-fused-attention.py")
    sys.exit(1)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "triton_fused_attention", TRITON_FUSED_ATTN_PATH
)
triton_fused_attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triton_fused_attention)
triton_attention = triton_fused_attention.attention

BATCH, N_HEADS = 4, 32

ALL_BACKENDS = []
if is_backend_available("cutile"):
    ALL_BACKENDS.append(("cutile", "CuTile", ("orange", "-")))
ALL_BACKENDS.append(("triton", "Triton", ("purple", "--")))

FLOPS_MULTIPLIER = {"fwd": 1.0, "bwd": 2.5, "fwd+bwd": 3.5}


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(datatype, HEAD_DIM, mode, causal):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(datatype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(8, 14)],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="TFLOPS",
        plot_name=(
            f"fused-attention-vs-triton-{mode}-batch{BATCH}-head{N_HEADS}"
            f"-d{HEAD_DIM}-causal={causal}-{dtype_name}-TFLOPS"
        ),
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "mode": mode,
            "causal": causal,
            "datatype": datatype,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(datatype, HEAD_DIM, mode, causal)
        for datatype in [torch.float16]
        for HEAD_DIM in [64, 128]
        for mode in ["fwd", "bwd", "fwd+bwd"]
        for causal in [True, False]
    ]
)
def bench_attention_vs_triton(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    mode,
    causal,
    backend,
    datatype,
    device=DEVICE,
):
    dtype = datatype
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )

    if backend == "cutile":

        def fwd_fn():
            return tile_fmha_with_backward(q, k, v, scaling=sm_scale, is_causal=causal)

    elif backend == "triton":

        def fwd_fn():
            return triton_attention(q, k, v, causal, sm_scale)

    else:
        return float("nan")

    try:
        if mode == "fwd":

            def fn():
                fwd_fn()

        elif mode == "bwd":
            o = fwd_fn()
            do = torch.randn_like(o)

            def fn():
                q.grad = k.grad = v.grad = None
                o.backward(do, retain_graph=True)

        else:  # fwd+bwd

            def fn():
                q.grad = k.grad = v.grad = None
                o = fwd_fn()
                o.backward(torch.randn_like(o))

        ms = triton.testing.do_bench(fn)
    except Exception:
        return float("nan")

    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    total_flops *= FLOPS_MULTIPLIER[mode]

    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_attention_vs_triton.run(print_data=True)
