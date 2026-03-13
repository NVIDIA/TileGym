#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import triton
import triton.testing
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel

from tilegym.backend import is_backend_available
from tilegym.ops.cutile.attention import tile_fmha_with_backward

DEVICE = triton.runtime.driver.active.get_active_torch_device()

BATCH, N_HEADS = 4, 32

# Build backend list based on availability
ALL_BACKENDS = []
if is_backend_available("cutile"):
    ALL_BACKENDS.append(("cutile", "CuTile", ("orange", "-")))
ALL_BACKENDS.append(("sdpa_flash", "SDPA-Flash", ("blue", "--")))
ALL_BACKENDS.append(("sdpa_memeff", "SDPA-MemEff", ("green", "-.")))
ALL_BACKENDS.append(("sdpa_math", "SDPA-Math", ("red", ":")))

SDPA_BACKEND_MAP = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_memeff": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
}

# FLOPs multiplier by mode.
# Forward: 2 matmuls (QK^T and PV).
# Backward: 5 matmuls (recompute QK^T, dV=P^T·dO, dP=dO·V^T, dQ=dS·K, dK=dS^T·Q) = 2.5x fwd.
# Combined: 3.5x fwd.
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
            f"fused-attention-train-{mode}-batch{BATCH}-head{N_HEADS}"
            f"-d{HEAD_DIM}-causal={causal}-{dtype_name}-TFLOPS-GBps"
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
def bench_fused_attention_backward(
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

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    # Build training forward function per backend (saves state for backward)
    if backend == "cutile":

        def fwd_fn():
            return tile_fmha_with_backward(q, k, v, scaling=sm_scale, is_causal=causal)

    elif backend in SDPA_BACKEND_MAP:
        sdp_b = SDPA_BACKEND_MAP[backend]

        def fwd_fn():
            with sdpa_kernel(sdp_b):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)

    else:
        return float("nan")

    try:
        if mode == "fwd":
            # Training forward (with saved LSE for backward)

            def fn():
                fwd_fn()

        elif mode == "bwd":
            # Pre-compute forward, then bench backward only
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

    # FLOPs: 2 matmuls (QK^T and PV), each 2*B*H*S*S*D
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    total_flops *= FLOPS_MULTIPLIER[mode]

    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    bench_fused_attention_backward.run(print_data=True)
