# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_group_gemm(
    group_A,
    group_B,
    static_persistent: bool = True,  # Unused - kept for interface compatibility
    use_tma: bool = True,  # Unused - kept for interface compatibility
    transpose_b: bool = False,
    **kwargs,
):
    """Reference implementation using PyTorch (loop over groups)."""
    if transpose_b:
        return [torch.matmul(A, B.transpose(-2, -1)) for A, B in zip(group_A, group_B)]
    return [torch.matmul(A, B) for A, B in zip(group_A, group_B)]


register_impl("group_gemm", "torch")(reference_group_gemm)


# Available backends with their display names and plot styles
ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def create_benchmark_config(dtype, group_size: int):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]

    # Keep ranges modest to avoid OOM in multi-group runs
    # (note: each data point allocates G*(A,B,C)).
    max_range = 14 if group_size >= 8 else 15  # 2^14..2^15 sized squares

    return triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2**i for i in range(10, max_range)],  # square GEMMs
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        xlabel="M/N/K",
        ylabel="TFLOPS",
        plot_name=f"group-gemm-g{group_size}-{dtype_name}-TFLOPS",
        args={"dtype": dtype, "group_size": group_size},
    )


configs = []
for dtype in [torch.float16, torch.bfloat16]:
    for group_size in [1, 2, 4, 8, 16]:
        cfg = create_benchmark_config(dtype, group_size)
        if cfg is not None:
            configs.append(cfg)


def _tflops_from_ms(ms: float, M: int, N: int, K: int, group_size: int) -> float:
    total_flops = group_size * (2 * M * N * K)
    return total_flops * 1e-12 / (ms * 1e-3)


def _measure_ms(fn, *, warmup: int = 5, rep: int = 20) -> float:
    # NOTE: do_bench_cudagraph can't be used with cuTile list args today.
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=None, return_mode="mean"))


def print_speedup_summary(
    *,
    dtypes=(torch.float16, torch.bfloat16),
    group_sizes=(1, 2, 4, 8, 16),
    group_size_ms=(1, 2, 4, 8, 16),
    m_exponents=range(10, 15),
    warmup: int = 5,
    rep: int = 20,
):
    """
    Print a lightweight speedup table (CuTile vs PyTorch).

    This is printed in addition to the standard triton perf_report output.
    """
    if not (is_backend_available("cutile")):
        print("\n[bench_group_gemm] CuTile backend unavailable; skipping speedup summary.")
        return

    print("\n" + "=" * 92)
    print("Speedup summary: CuTile / PyTorch  (computed from mean-ms timings; higher is better)")
    print("=" * 92)

    for dtype in dtypes:
        dtype_name = str(dtype).split(".")[-1]
        print(f"\n--- dtype={dtype_name} ---")
        for g in group_sizes:
            # Match the perf_report sizing logic (smaller upper bound for larger groups).
            max_exp = 14 if g >= 8 else 15
            exps = [e for e in m_exponents if e < max_exp]

            print(f"\n[group_size={g}]")
            header = (
                f"{'M=N=K':>10}  "
                f"{'TFLOPS_torch':>12}  "
                f"{'TFLOPS_cutile(best)':>18}  "
                f"{'GROUP_SIZE_M':>12}  "
                f"{'speedup':>9}"
            )
            print(header)
            print("-" * len(header))

            for e in exps:
                M = N = K = 2**e
                group_A = [torch.randn((M, K), device=DEVICE, dtype=dtype) for _ in range(g)]
                group_B = [torch.randn((K, N), device=DEVICE, dtype=dtype) for _ in range(g)]

                fn_torch = lambda: reference_group_gemm(group_A, group_B, transpose_b=False)
                ms_torch = _measure_ms(fn_torch, warmup=warmup, rep=rep)
                tflops_torch = _tflops_from_ms(ms_torch, M, N, K, g)

                # Sweep GROUP_SIZE_M and take the best cuTile throughput.
                best_tflops_cutile = -1.0
                best_group_size_m = None
                for gsm in group_size_ms:
                    fn_cutile = lambda gsm=gsm: tilegym.ops.group_gemm(
                        group_A,
                        group_B,
                        static_persistent=True,
                        use_tma=True,
                        transpose_b=False,
                        backend="cutile",
                        kernel_configs={"GROUP_SIZE_M": int(gsm)},
                    )
                    ms_cutile = _measure_ms(fn_cutile, warmup=warmup, rep=rep)
                    tflops_cutile = _tflops_from_ms(ms_cutile, M, N, K, g)
                    if tflops_cutile > best_tflops_cutile:
                        best_tflops_cutile = tflops_cutile
                        best_group_size_m = int(gsm)

                speedup = best_tflops_cutile / max(tflops_torch, 1e-9)
                print(
                    f"{M:10d}  "
                    f"{tflops_torch:12.2f}  "
                    f"{best_tflops_cutile:18.2f}  "
                    f"{best_group_size_m:12d}  "
                    f"{speedup:9.2f}x"
                )


@triton.testing.perf_report(configs)
def benchmark(M, N, K, backend, dtype, group_size):
    # Build a homogeneous group of GEMMs: (M,K) @ (K,N)
    group_A = [torch.randn((M, K), device=DEVICE, dtype=dtype) for _ in range(group_size)]
    group_B = [torch.randn((K, N), device=DEVICE, dtype=dtype) for _ in range(group_size)]

    quantiles = [0.5, 0.2, 0.8]

    fn = lambda: tilegym.ops.group_gemm(
        group_A,
        group_B,
        static_persistent=True,
        use_tma=True,
        transpose_b=False,
        backend=backend,
    )

    # Quick correctness check (small-ish only to avoid spending too much time in validation)
    if M <= 2048 and backend != "torch":
        out = fn()
        ref = reference_group_gemm(group_A, group_B)
        for o, r in zip(out, ref):
            torch.testing.assert_close(o, r, atol=2e-2, rtol=2e-2)

    # cuTile group_gemm currently passes Python lists into ct.launch, which is not
    # compatible with CUDA Graph capture. Use non-cudagraph benchmarking.
    #
    # If/when list arguments become graph-capture-friendly, we can switch back to
    # do_bench_cudagraph for slightly lower measurement noise.
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    perf = lambda t_ms: _tflops_from_ms(float(t_ms), M, N, K, group_size)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
    print_speedup_summary()
