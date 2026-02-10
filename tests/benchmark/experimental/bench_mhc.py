# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import triton

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

X_DTYPE = torch.bfloat16
W_DTYPE = torch.float32
OTHER_DTYPE = torch.float32


def _is_fp8_dtype(dtype):
    return "float8" in str(dtype)


def _get_benchmark_dtypes():
    dtypes = [X_DTYPE]
    return dtypes


def _randn(shape, dtype, device):
    if _is_fp8_dtype(dtype):
        return torch.randn(shape, device=device, dtype=torch.float16).to(dtype)
    return torch.randn(shape, device=device, dtype=dtype)


BENCH_DTYPES = _get_benchmark_dtypes()


def reference_mhc_apply_residual(
    x: torch.Tensor,
    f_out: torch.Tensor,
    y: torch.Tensor,
    n: int,
):
    B, nC = x.shape
    C = nC // n
    x_view = x.view(B, n, C)
    f_view = f_out
    h_post = y[:, n : 2 * n]
    h_res = y[:, 2 * n : 2 * n + n * n].view(B, n, n)
    x_res = torch.matmul(h_res, x_view)
    x_post = h_post.unsqueeze(-1) * f_view.unsqueeze(1)
    x_next = x_res + x_post
    return x_next.to(x.dtype).view(B, nC)


def reference_mhc_gemm_rms_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    w_nt: torch.Tensor = None,
):
    x_fp32 = x.to(torch.float32)
    if w_nt is not None:
        w_nt_fp32 = w_nt.to(torch.float32)
        y = x_fp32 @ w_nt_fp32.transpose(0, 1)
    else:
        w_fp32 = w.to(torch.float32)
        y = x_fp32 @ w_fp32
    rms = torch.sqrt(x_fp32.pow(2).mean(dim=1, keepdim=True))
    scale = torch.empty((y.shape[1],), dtype=torch.float32, device=y.device)
    scale[:n] = alpha_pre
    scale[n : 2 * n] = alpha_post
    scale[2 * n :] = alpha_res
    linear = y * scale / rms + bias.to(torch.float32)
    out = linear.clone()
    out[:, :n] = torch.sigmoid(linear[:, :n])
    out[:, n : 2 * n] = 2.0 * torch.sigmoid(linear[:, n : 2 * n])
    return out.to(bias.dtype), rms


def reference_mhc_sinkhorn(
    y: torch.Tensor,
    n: int,
):
    start = 2 * n
    end = start + n * n
    mat = y[:, start:end].to(torch.float32).reshape(-1, n, n)
    mat = torch.exp(mat)
    for _ in range(20):
        mat = mat / mat.sum(dim=2, keepdim=True)
        mat = mat / mat.sum(dim=1, keepdim=True)
    y[:, start:end] = mat.reshape(-1, n * n).to(y.dtype)
    return y


register_impl("mhc_apply_residual", "torch")(reference_mhc_apply_residual)
register_impl("mhc_gemm_rms_scale", "torch")(reference_mhc_gemm_rms_scale)
register_impl("mhc_sinkhorn", "torch")(reference_mhc_sinkhorn)


ALL_BACKENDS = [
    ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]

# Try to add deepgemm backend if available
try:
    import deep_gemm

    DEEPGEMM_BACKEND = ("deepgemm", "DeepGemm", ("blue", "-."))
    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_BACKEND = None
    DEEPGEMM_AVAILABLE = False


def get_supported_backends():
    return [p for p in ALL_BACKENDS if p is not None]


def get_supported_backends_split_gemm_rms():
    backends = [
        ("cutile", "CuTile", ("orange", "-")) if is_backend_available("cutile") else None,
        ("torch", "PyTorch", ("green", "-")),
        DEEPGEMM_BACKEND if DEEPGEMM_AVAILABLE else None,
    ]
    return [p for p in backends if p is not None]


def create_gemm_rms_scale_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    n = 4
    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[8192],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mhc-gemm-rms-scale-performance-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "K": n * 7168,
            "N": n * n + 2 * n,
            "n": n,
        },
    )


def create_sinkhorn_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    n = 4
    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[8192],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mhc-sinkhorn-performance-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "N": n * n + 2 * n,
            "n": n,
        },
    )


def create_apply_residual_benchmark_config(dtype):
    available_backends = get_supported_backends()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    n = 4
    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[8192],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mhc-apply-residual-performance-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "C": 7168,
            "n": n,
        },
    )


def create_split_gemm_rms_benchmark_config(dtype):
    available_backends = get_supported_backends_split_gemm_rms()
    if not available_backends:
        return None

    backends, names, styles = zip(*available_backends)
    dtype_name = str(dtype).split(".")[-1]
    n = 4
    return triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[8192],
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"mhc-split-gemm-rms-performance-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "K": n * 7168,
            "N": n * n + 2 * n,
            "n": n,
        },
        y_log=False,
    )


@triton.testing.perf_report([create_split_gemm_rms_benchmark_config(dtype) for dtype in BENCH_DTYPES])
def bench_mhc_split_gemm_rms(M, backend, dtype, K, N, n, device=DEVICE):
    """Benchmark for split GEMM + RMS kernel only (compared with deepgemm)"""
    x = _randn((M, K), dtype=X_DTYPE, device=device)
    w = _randn((K, N), dtype=W_DTYPE, device=device)
    w_nt = w.transpose(0, 1).contiguous()

    cfg = None
    if is_backend_available("cutile"):
        from tilegym.ops.cutile.experimental import mhc as mhc_cutile

        _, _, cfg = mhc_cutile.cutile_autotune_mhc_split_gemm_rms(
            torch.cuda.current_stream(),
            x,
            w,
            M,
            N,
            K,
        )

    if backend == "deepgemm":
        if not DEEPGEMM_AVAILABLE:
            return 0.0

        import deep_gemm

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        split_k = 4
        dg_d = torch.empty((split_k, M, N), dtype=torch.float32, device=device)
        dg_s = torch.empty((split_k, M), dtype=torch.float32, device=device)

        fn = lambda: deep_gemm.tf32_hc_prenorm_gemm(x, w_nt, dg_d, dg_s, num_splits=split_k)
        ms = triton.testing.do_bench_cudagraph(fn)

        # Output bytes: dg_d and dg_s (split outputs)
        x_bytes = M * K * x.element_size()
        w_bytes = K * N * w.element_size()
        out_d_bytes = split_k * M * N * 4  # float32
        out_s_bytes = split_k * M * 4  # float32
        total_bytes = x_bytes + w_bytes + out_d_bytes + out_s_bytes
        gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
        return gb_per_s

    if backend == "torch":
        if cfg is None:
            split_k = 4
            tile_size_n = 128
        else:
            split_k = cfg.SPLIT_K
            tile_size_n = cfg.TILE_SIZE_N

        num_bid_n = triton.cdiv(N, tile_size_n)
        y_acc = torch.empty((split_k, M, N), device=x.device, dtype=torch.float32)
        r_acc = torch.empty((split_k, M, num_bid_n), device=x.device, dtype=torch.float32)

        k_per_split = triton.cdiv(K, split_k)
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        def run():
            for s in range(split_k):
                k_start = s * k_per_split
                k_end = min(k_start + k_per_split, K)
                x_s = x_fp32[:, k_start:k_end]
                w_s = w_fp32[k_start:k_end, :]
                y_acc[s].copy_(x_s @ w_s)
                rms_partial = (x_s * x_s).sum(dim=1, keepdim=True)  # [M, 1]
                r_acc[s].copy_(rms_partial.expand(M, num_bid_n))

        ms = triton.testing.do_bench_cudagraph(run)
        total_bytes = M * K * x.element_size() + K * N * w.element_size() + y_acc.numel() * 4 + r_acc.numel() * 4
        gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
        return gb_per_s

    elif backend == "cutile":
        import cuda.tile as ct

        from tilegym.ops.cutile.experimental import mhc as mhc_cutile

        y_acc = torch.empty((M * cfg.SPLIT_K, N), device=x.device, dtype=torch.float32)
        r_acc = torch.empty((M * cfg.SPLIT_K, triton.cdiv(N, cfg.TILE_SIZE_N)), device=x.device, dtype=torch.float32)

        grid = (
            triton.cdiv(M, cfg.TILE_SIZE_M) * triton.cdiv(N, cfg.TILE_SIZE_N),
            cfg.SPLIT_K,
            1,
        )
        fn = lambda: ct.launch(
            torch.cuda.current_stream(),
            grid,
            mhc_cutile.mhc_split_gemm_rms_kernel,
            (
                x,
                w,
                y_acc,
                r_acc,
                M,
                N,
                K,
                cfg.TILE_SIZE_M,
                cfg.TILE_SIZE_N,
                cfg.TILE_SIZE_K,
                cfg.SPLIT_K,
                cfg.GROUP_SIZE_M,
            ),
        )

        ms = triton.testing.do_bench_cudagraph(fn)

        # Calculate bandwidth
        x_bytes = M * K * x.element_size()
        w_bytes = K * N * w.element_size()
        y_acc_bytes = M * cfg.SPLIT_K * N * 4  # float32
        r_acc_bytes = r_acc.numel() * 4  # float32
        total_bytes = x_bytes + w_bytes + y_acc_bytes + r_acc_bytes
        gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
        return gb_per_s

    return 0.0


@triton.testing.perf_report([create_gemm_rms_scale_benchmark_config(dtype) for dtype in BENCH_DTYPES])
def bench_mhc_gemm_rms_scale(M, backend, dtype, K, N, n, device=DEVICE):
    """Benchmark for full MHC GEMM+RMS+Scale operation (compared with torch)"""
    x = _randn((M, K), dtype=dtype, device=device)
    w = _randn((K, N), dtype=W_DTYPE, device=device)
    bias = _randn((N,), dtype=OTHER_DTYPE, device=device)
    alpha_pre = 0.8
    alpha_post = 1.1
    alpha_res = 0.9

    kwargs = {}
    if backend == "cutile":
        from tilegym.ops.cutile.experimental import mhc as mhc_cutile

        _, _, cfg = mhc_cutile.cutile_autotune_mhc_split_gemm_rms(
            torch.cuda.current_stream(),
            x,
            w,
            M,
            N,
            K,
        )
        kwargs["cfg"] = cfg

    fn = lambda: tilegym.ops.mhc_gemm_rms_scale(
        x, w, n, alpha_pre, alpha_post, alpha_res, bias, backend=backend, **kwargs
    )

    ms = triton.testing.do_bench_cudagraph(fn)

    # Calculate bandwidth (GB/s)
    # Input: x (M*K), w (K*N), bias (N)
    # Output: y (M*N), r (M*1)
    x_bytes = M * K * x.element_size()
    w_bytes = K * N * w.element_size()
    bias_bytes = N * bias.element_size()
    y_bytes = M * N * bias.element_size()  # output dtype same as bias
    r_bytes = M * 1 * 4  # r is float32
    total_bytes = x_bytes + w_bytes + bias_bytes + y_bytes + r_bytes
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


@triton.testing.perf_report([create_sinkhorn_benchmark_config(dtype) for dtype in BENCH_DTYPES])
def bench_mhc_sinkhorn(M, backend, dtype, N, n, device=DEVICE):
    """Benchmark for MHC Sinkhorn operation (compared with torch)"""
    y = _randn((M, N), dtype=dtype, device=device)

    y_test = y.clone()
    tilegym.ops.mhc_sinkhorn(y_test, n, backend=backend)

    fn = lambda: tilegym.ops.mhc_sinkhorn(y_test, n, backend=backend)
    ms = triton.testing.do_bench_cudagraph(fn)

    bytes_per_row = n * n * y.element_size()
    total_bytes = y.shape[0] * bytes_per_row * 2
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


@triton.testing.perf_report([create_apply_residual_benchmark_config(dtype) for dtype in BENCH_DTYPES])
def bench_mhc_apply_residual(M, backend, dtype, C, n, device=DEVICE):
    """Benchmark for MHC apply residual operation (compared with torch)"""
    nC = n * C
    out_n = n * n + 2 * n
    x = _randn((M, nC), dtype=dtype, device=device)
    f_out = _randn((M, C), dtype=dtype, device=device)
    y = _randn((M, out_n), dtype=dtype, device=device)

    fn = lambda: tilegym.ops.mhc_apply_residual(x, f_out, y, n, backend=backend)
    out = fn()
    ms = triton.testing.do_bench_cudagraph(fn)

    total_bytes = (
        x.numel() * x.element_size()
        + f_out.numel() * f_out.element_size()
        + y.numel() * y.element_size()
        + out.numel() * out.element_size()
    )
    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_mhc_split_gemm_rms.run(print_data=True)
    bench_mhc_gemm_rms_scale.run(print_data=True)
    bench_mhc_sinkhorn.run(print_data=True)
    bench_mhc_apply_residual.run(print_data=True)
