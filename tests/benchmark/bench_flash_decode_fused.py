# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available

DEVICE = triton.runtime.driver.active.get_active_torch_device()


ALL_IMPLS = [
    ("two_kernel", "CuTile (2-kernel)", ("orange", "-")),
    ("fused", "CuTile (fused)", ("blue", "-")),
]


def _tflops(ms: float, B: int, H_q: int, S_kv: int, D: int) -> float:
    # Rough FLOP model for single-token decode:
    # - QK: 2 * B * H_q * S_kv * D
    # - PV: 2 * B * H_q * S_kv * D
    # total ~ 4 * B * H_q * S_kv * D
    flops = 4 * B * H_q * S_kv * D
    return flops * 1e-12 / (ms * 1e-3)


def create_benchmark_config(dtype: torch.dtype, group_size: int):
    dtype_name = str(dtype).split(".")[-1]
    return triton.testing.Benchmark(
        x_names=["S_kv"],
        x_vals=[9, 119, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="impl",
        line_vals=[i for (i, _, _) in ALL_IMPLS],
        line_names=[n for (_, n, _) in ALL_IMPLS],
        styles=[s for (_, _, s) in ALL_IMPLS],
        xlabel="S_kv",
        ylabel="TFLOPS",
        plot_name=f"flash-decode-fused-vs-2kernel-g{group_size}-{dtype_name}-TFLOPS",
        args={"dtype": dtype, "group_size": group_size},
    )


configs = [create_benchmark_config(torch.float16, g) for g in [1, 4, 8]]


@triton.testing.perf_report(configs)
def benchmark(S_kv: int, impl: str, dtype: torch.dtype, group_size: int):
    if not is_backend_available("cutile"):
        raise RuntimeError("cutile backend unavailable")
    tilegym.set_backend("cutile")

    # Match test defaults
    B = 2
    H_q = 32
    D = 64
    H_kv = H_q // group_size
    sm_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    q = torch.randn(B, H_q, 1, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H_kv, S_kv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H_kv, S_kv, D, device=DEVICE, dtype=dtype)

    if impl == "two_kernel":
        fn = lambda: tilegym.ops.fmha_decode(q=q, k=k, v=v, sm_scale=sm_scale)
    elif impl == "fused":
        fn = lambda: tilegym.ops.fmha_decode_fused(q=q, k=k, v=v, sm_scale=sm_scale)
    else:
        raise ValueError(f"Unknown impl: {impl}")

    # Lightweight correctness check vs baseline for smaller sizes
    if S_kv <= 512 and impl == "fused":
        ref = tilegym.ops.fmha_decode(q=q, k=k, v=v, sm_scale=sm_scale)
        out = fn()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    # NOTE: we use non-cudagraph timing here because these ops allocate internal
    # workspaces; cudagraph capture may fail depending on allocator behavior.
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    perf = lambda t: _tflops(float(t), B, H_q, S_kv, D)
    return perf(ms), perf(max_ms), perf(min_ms)


def print_speedup_table():
    if not is_backend_available("cutile"):
        print("\n[bench_flash_decode_fused] CuTile backend unavailable; skipping speedup table.")
        return
    tilegym.set_backend("cutile")

    B, H_q, D = 2, 32, 64
    sm_scale = 1.0 / math.sqrt(D)

    print("\n" + "=" * 84)
    print("Speedup: fused / two-kernel  (computed from mean-ms timings; higher is better)")
    print("=" * 84)

    for group_size in [1, 4, 8]:
        H_kv = H_q // group_size
        print(f"\n[group_size={group_size}]")
        header = f"{'S_kv':>6}  {'TFLOPS_2k':>12}  {'TFLOPS_fused':>13}  {'speedup':>9}"
        print(header)
        print("-" * len(header))

        for S_kv in [9, 119, 256, 512, 1024, 2048, 4096, 8192]:
            torch.manual_seed(0)
            q = torch.randn(B, H_q, 1, D, device=DEVICE, dtype=torch.float16)
            k = torch.randn(B, H_kv, S_kv, D, device=DEVICE, dtype=torch.float16)
            v = torch.randn(B, H_kv, S_kv, D, device=DEVICE, dtype=torch.float16)

            fn_2k = lambda: tilegym.ops.fmha_decode(q=q, k=k, v=v, sm_scale=sm_scale)
            fn_fused = lambda: tilegym.ops.fmha_decode_fused(q=q, k=k, v=v, sm_scale=sm_scale)

            ms_2k = float(triton.testing.do_bench(fn_2k, warmup=5, rep=20, return_mode="mean"))
            ms_fused = float(triton.testing.do_bench(fn_fused, warmup=5, rep=20, return_mode="mean"))

            t2 = _tflops(ms_2k, B, H_q, S_kv, D)
            tf = _tflops(ms_fused, B, H_q, S_kv, D)
            sp = tf / max(t2, 1e-9)
            print(f"{S_kv:6d}  {t2:12.2f}  {tf:13.2f}  {sp:9.2f}x")


if __name__ == "__main__":
    benchmark.run(print_data=True)
    print_speedup_table()

