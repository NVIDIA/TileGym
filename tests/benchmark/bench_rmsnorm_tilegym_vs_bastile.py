"""
RMSNorm Benchmark: TileGym vs Bastile vs PyTorch
=================================================
Compares the CuTile RMSNorm implementations from TileGym (original) and
Bastile (optimised persistent kernels), plus raw PyTorch, across a range
of (M, N) shapes relevant to Qwen3-8B (N = 4096) and other models.

Reports latency (µs), bandwidth (GB/s), and speedup for:
  - Forward only
  - Backward only (dx + dw)
  - Forward + Backward (full autograd)
"""

import torch
import time

# ── TileGym ──────────────────────────────────────────────────────────────
from tilegym.ops.cutile.rms_norm import (
    RMSNorm as TileGymRMSNormFn,      # autograd Function
    rms_norm_backward as tilegym_bwd,
    TileRMSNorm,                       # Module (for compute_rstd_torch)
)

# ── Bastile ──────────────────────────────────────────────────────────────
from bastile.ops.rms_norm import (
    CuTileRMSNormFunction as BastileRMSNormFn,
    rms_norm as bastile_rms_norm,
)

# ── Constants ────────────────────────────────────────────────────────────
WARMUP = 10
BENCH = 100
DEVICE = "cuda"

# Qwen3-8B-relevant shapes: M = batch*seq_len, N = hidden_size
CONFIGS = [
    # (M,    N)
    (256,    4096),
    (512,    4096),
    (1024,   4096),
    (2048,   4096),
    (4096,   4096),
    (8192,   4096),
    (16384,  4096),
    # Extra hidden sizes for generality
    (4096,   2048),
    (4096,   5120),
    (4096,   8192),
]

DTYPE = torch.bfloat16


# ── Helpers ──────────────────────────────────────────────────────────────
def pytorch_rms_norm(x, weight, eps):
    """Raw PyTorch RMSNorm."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def benchmark_fn(fn, warmup=WARMUP, iters=BENCH):
    """Return median latency in seconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    times.sort()
    # median
    mid = len(times) // 2
    return times[mid] * 1e-3  # → seconds


def gbps(total_bytes, latency_s):
    return total_bytes / latency_s * 1e-9


# ── Forward benchmark ────────────────────────────────────────────────────
def bench_forward(M, N, dtype=DTYPE):
    eps = 1e-6
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    w = torch.ones(N, device=DEVICE, dtype=dtype)

    # bytes: read x + read w + write y
    bpe = x.element_size()
    total_bytes = (M * N * bpe) + (N * bpe) + (M * N * bpe)

    # TileGym (static_persistent=True to match Bastile's approach)
    fn_tg_sp = lambda: TileGymRMSNormFn.apply(x, None, w, eps, None, True, 0.0)
    # TileGym gather
    fn_tg_g = lambda: TileGymRMSNormFn.apply(x, None, w, eps, None, False, 0.0)
    # Bastile persistent
    fn_ba = lambda: bastile_rms_norm(x, w, eps)
    # PyTorch
    fn_pt = lambda: pytorch_rms_norm(x, w, eps)

    lat_tg_sp = benchmark_fn(fn_tg_sp)
    lat_tg_g = benchmark_fn(fn_tg_g)
    lat_ba = benchmark_fn(fn_ba)
    lat_pt = benchmark_fn(fn_pt)

    return {
        "TileGym-Persist": (lat_tg_sp, gbps(total_bytes, lat_tg_sp)),
        "TileGym-Gather":  (lat_tg_g,  gbps(total_bytes, lat_tg_g)),
        "Bastile":         (lat_ba,    gbps(total_bytes, lat_ba)),
        "PyTorch":         (lat_pt,    gbps(total_bytes, lat_pt)),
    }


# ── Backward benchmark ──────────────────────────────────────────────────
def bench_backward(M, N, dtype=DTYPE):
    eps = 1e-6
    x = torch.randn(M, N, device=DEVICE, dtype=dtype)
    w = torch.ones(N, device=DEVICE, dtype=dtype)
    dy = torch.randn(M, N, device=DEVICE, dtype=dtype)

    # Compute rstd via PyTorch for a fair comparison
    rstd = TileRMSNorm.compute_rstd_torch(x, eps)

    # bytes: read x + read dy + read w + read rstd + write dx + write dw + temp_buffer read/write
    bpe = x.element_size()
    total_bytes = (M * N * bpe) * 2 + (N * bpe) * 2 + (M * 4) + (M * N * 4) * 2  # approx

    # TileGym backward (standalone, no autograd)
    fn_tg = lambda: tilegym_bwd(x, dy, w, rstd)

    # Bastile backward via autograd (need to run forward first to get rstd)
    # We'll use the full autograd path to be fair
    def fn_ba():
        x_r = x.detach().requires_grad_(True)
        w_r = w.detach().requires_grad_(True)
        out = bastile_rms_norm(x_r, w_r, eps)
        out.backward(dy)
        return x_r.grad, w_r.grad

    # PyTorch backward via autograd
    def fn_pt():
        x_r = x.detach().requires_grad_(True)
        w_r = w.detach().requires_grad_(True)
        out = pytorch_rms_norm(x_r, w_r, eps)
        out.backward(dy)
        return x_r.grad, w_r.grad

    lat_tg = benchmark_fn(fn_tg)
    lat_ba = benchmark_fn(fn_ba)
    lat_pt = benchmark_fn(fn_pt)

    return {
        "TileGym":  (lat_tg, gbps(total_bytes, lat_tg)),
        "Bastile":  (lat_ba, gbps(total_bytes, lat_ba)),
        "PyTorch":  (lat_pt, gbps(total_bytes, lat_pt)),
    }


# ── Forward + Backward benchmark ────────────────────────────────────────
def bench_fwd_bwd(M, N, dtype=DTYPE):
    eps = 1e-6
    x_base = torch.randn(M, N, device=DEVICE, dtype=dtype)
    w_base = torch.ones(N, device=DEVICE, dtype=dtype)

    def fn_tg():
        x_r = x_base.detach().requires_grad_(True)
        w_r = w_base.detach().requires_grad_(True)
        # TileGym only supports backward with gather mode (static_persistent=False)
        out = TileGymRMSNormFn.apply(x_r, None, w_r, eps, None, False, 0.0)
        out.sum().backward()

    def fn_ba():
        x_r = x_base.detach().requires_grad_(True)
        w_r = w_base.detach().requires_grad_(True)
        out = bastile_rms_norm(x_r, w_r, eps)
        out.sum().backward()

    def fn_pt():
        x_r = x_base.detach().requires_grad_(True)
        w_r = w_base.detach().requires_grad_(True)
        out = pytorch_rms_norm(x_r, w_r, eps)
        out.sum().backward()

    lat_tg = benchmark_fn(fn_tg)
    lat_ba = benchmark_fn(fn_ba)
    lat_pt = benchmark_fn(fn_pt)

    return {
        "TileGym":  lat_tg,
        "Bastile":  lat_ba,
        "PyTorch":  lat_pt,
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 100)
    print("RMSNorm Benchmark: TileGym vs Bastile vs PyTorch")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Dtype:  {DTYPE}")
    print(f"  Warmup: {WARMUP}  Iters: {BENCH}")
    print("=" * 100)

    # ── Warmup JIT ───────────────────────────────────────────────────────
    print("\nJIT warmup...")
    for M, N in [(256, 4096), (4096, 4096)]:
        x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
        w = torch.ones(N, device=DEVICE, dtype=DTYPE, requires_grad=True)
        # TileGym static_persistent (forward only — no backward support)
        _ = TileGymRMSNormFn.apply(x.detach(), None, w.detach(), 1e-6, None, True, 0.0)
        # TileGym gather (supports backward)
        x_g = x.detach().requires_grad_(True)
        w_g = w.detach().requires_grad_(True)
        out = TileGymRMSNormFn.apply(x_g, None, w_g, 1e-6, None, False, 0.0)
        out.sum().backward()
        # Bastile
        x2 = x.detach().requires_grad_(True)
        w2 = w.detach().requires_grad_(True)
        out = bastile_rms_norm(x2, w2, 1e-6)
        out.sum().backward()
    torch.cuda.synchronize()
    print("JIT warmup done.\n")

    # ── Forward Benchmark ────────────────────────────────────────────────
    print("─" * 100)
    print("FORWARD ONLY")
    print("─" * 100)
    header = f"{'M':>7} {'N':>6} │ {'TG-Persist µs':>14} {'GB/s':>7} │ {'TG-Gather µs':>13} {'GB/s':>7} │ {'Bastile µs':>11} {'GB/s':>7} │ {'PyTorch µs':>11} {'GB/s':>7} │ {'Best':>12}"
    print(header)
    print("─" * len(header))

    for M, N in CONFIGS:
        res = bench_forward(M, N)
        tg_sp_lat, tg_sp_bw = res["TileGym-Persist"]
        tg_g_lat, tg_g_bw = res["TileGym-Gather"]
        ba_lat, ba_bw = res["Bastile"]
        pt_lat, pt_bw = res["PyTorch"]

        lats = {"TG-Persist": tg_sp_lat, "TG-Gather": tg_g_lat, "Bastile": ba_lat, "PyTorch": pt_lat}
        best = min(lats, key=lats.get)
        print(f"{M:>7} {N:>6} │ {tg_sp_lat*1e6:>11.1f} µs {tg_sp_bw:>7.0f} │ {tg_g_lat*1e6:>10.1f} µs {tg_g_bw:>7.0f} │ {ba_lat*1e6:>8.1f} µs {ba_bw:>7.0f} │ {pt_lat*1e6:>8.1f} µs {pt_bw:>7.0f} │ {best:>12}")

    # ── Backward Benchmark ───────────────────────────────────────────────
    print()
    print("─" * 100)
    print("BACKWARD (fwd+bwd via autograd for Bastile/PyTorch; standalone kernel for TileGym)")
    print("─" * 100)
    header = f"{'M':>7} {'N':>6} │ {'TileGym µs':>11} {'GB/s':>7} │ {'Bastile µs':>11} {'GB/s':>7} │ {'PyTorch µs':>11} {'GB/s':>7} │ {'Best':>12}"
    print(header)
    print("─" * len(header))

    for M, N in CONFIGS:
        res = bench_backward(M, N)
        tg_lat, tg_bw = res["TileGym"]
        ba_lat, ba_bw = res["Bastile"]
        pt_lat, pt_bw = res["PyTorch"]

        lats = {"TileGym": tg_lat, "Bastile": ba_lat, "PyTorch": pt_lat}
        best = min(lats, key=lats.get)
        print(f"{M:>7} {N:>6} │ {tg_lat*1e6:>8.1f} µs {tg_bw:>7.0f} │ {ba_lat*1e6:>8.1f} µs {ba_bw:>7.0f} │ {pt_lat*1e6:>8.1f} µs {pt_bw:>7.0f} │ {best:>12}")

    # ── Fwd+Bwd Benchmark ────────────────────────────────────────────────
    print()
    print("─" * 100)
    print("FORWARD + BACKWARD (full autograd)")
    print("─" * 100)
    header = f"{'M':>7} {'N':>6} │ {'TileGym µs':>11} │ {'Bastile µs':>11} │ {'PyTorch µs':>11} │ {'Best':>12} │ {'Ba/TG':>6} {'Ba/PT':>6}"
    print(header)
    print("─" * len(header))

    for M, N in CONFIGS:
        res = bench_fwd_bwd(M, N)
        tg_lat = res["TileGym"]
        ba_lat = res["Bastile"]
        pt_lat = res["PyTorch"]

        lats = {"TileGym": tg_lat, "Bastile": ba_lat, "PyTorch": pt_lat}
        best = min(lats, key=lats.get)
        ba_vs_tg = ba_lat / tg_lat
        ba_vs_pt = ba_lat / pt_lat
        print(f"{M:>7} {N:>6} │ {tg_lat*1e6:>8.1f} µs │ {ba_lat*1e6:>8.1f} µs │ {pt_lat*1e6:>8.1f} µs │ {best:>12} │ {ba_vs_tg:>5.2f}x {ba_vs_pt:>5.2f}x")

    print()
    print("=" * 100)
    print("Ba/TG = Bastile / TileGym ratio (<1 = Bastile faster)")
    print("Ba/PT = Bastile / PyTorch ratio (<1 = Bastile faster)")
    print("=" * 100)


if __name__ == "__main__":
    main()
