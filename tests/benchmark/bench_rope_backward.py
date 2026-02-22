# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark RoPE backward and full pass performance."""

import torch
import triton
import triton.testing
from bench_rope import DEVICE
from bench_rope import apply_rope_torch
from bench_rope import create_rotary_embeddings
from bench_rope import get_supported_backends

import tilegym


def create_benchmark_config(mode: str, dtype: torch.dtype, bsz: int, num_heads: int, head_dim: int):
    providers = get_supported_backends(dtype)
    if not providers:
        return None

    backends, names, styles = zip(*providers)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[2**i for i in range(10, 15)],  # 1024 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rope-{mode}-bsz{bsz}-heads{num_heads}-d{head_dim}-{dtype_name}-GBps",
        args={
            "BSZ": bsz,
            "NUM_HEADS": num_heads,
            "HEAD_DIM": head_dim,
            "datatype": dtype,
            "mode": mode,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(mode, dtype, bsz, num_heads, head_dim)
        for mode in ["backward", "full"]
        for dtype in [torch.float16]
        for bsz in [1]
        for num_heads in [16]
        for head_dim in [64]
    ]
)
def bench_rope_backward(
    BSZ,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    backend,
    datatype,
    mode,
    device=DEVICE,
):
    dtype = datatype

    # Correctness check on a tiny config first.
    q_ref = torch.randn((1, NUM_HEADS, 128, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k_ref = torch.randn((1, NUM_HEADS, 128, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    q_opt = q_ref.detach().clone().requires_grad_(True)
    k_opt = k_ref.detach().clone().requires_grad_(True)

    pos_ids_ref = torch.arange(128, device=device, dtype=torch.long).unsqueeze(0)
    cos_ref, sin_ref = create_rotary_embeddings(128, HEAD_DIM, dtype, device)
    cos_ref = cos_ref.unsqueeze(0)
    sin_ref = sin_ref.unsqueeze(0)

    oq_t, ok_t = apply_rope_torch(q_ref, k_ref, cos_ref, sin_ref, pos_ids_ref)
    gq_t = torch.randn_like(oq_t)
    gk_t = torch.randn_like(ok_t)
    (oq_t * gq_t + ok_t * gk_t).sum().backward()

    oq_c, ok_c = tilegym.ops.apply_rope_base(q_opt, k_opt, cos_ref, sin_ref, pos_ids_ref, backend=backend)
    (oq_c * gq_t + ok_c * gk_t).sum().backward()

    torch.testing.assert_close(oq_c, oq_t, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ok_c, ok_t, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_opt.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_opt.grad, k_ref.grad, atol=2e-2, rtol=2e-2)

    # Benchmark tensors.
    q_base = torch.randn((BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
    k_base = torch.randn((BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
    pos_ids = torch.arange(SEQ_LEN, device=device, dtype=torch.long).unsqueeze(0).expand(BSZ, -1)
    cos, sin = create_rotary_embeddings(SEQ_LEN, HEAD_DIM, dtype, device)
    cos = cos.unsqueeze(0).expand(BSZ, -1, -1)
    sin = sin.unsqueeze(0).expand(BSZ, -1, -1)

    bytes_per_element = q_base.element_size()
    qk_elems = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM
    cos_sin_elems = 2 * BSZ * SEQ_LEN * HEAD_DIM

    if mode == "backward":
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        out_q, out_k = tilegym.ops.apply_rope_base(q, k, cos, sin, pos_ids, backend=backend)
        grad_q = torch.randn_like(out_q)
        grad_k = torch.randn_like(out_k)

        def bwd_only():
            torch.autograd.backward((out_q, out_k), (grad_q, grad_k), retain_graph=True)

        ms = triton.testing.do_bench(bwd_only, rep=10)
        # Approximate backward traffic: read dq/dk + read cos/sin + read/write q/k grads.
        total_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
    else:

        def full_pass():
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            out_q, out_k = tilegym.ops.apply_rope_base(q, k, cos, sin, pos_ids, backend=backend)
            grad_q = torch.randn_like(out_q)
            grad_k = torch.randn_like(out_k)
            (out_q * grad_q + out_k * grad_k).sum().backward()

        ms = triton.testing.do_bench(full_pass, rep=10)
        # Forward + backward approximate total traffic.
        fwd_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
        bwd_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
        total_bytes = fwd_bytes + bwd_bytes

    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_rope_backward.run(print_data=True)
