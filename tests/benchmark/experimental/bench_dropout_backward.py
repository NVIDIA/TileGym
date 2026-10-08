# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Dropout autograd benchmark: backward alone and forward + backward.

Uses warmed CUDA Graph replay, including contiguous upstream gradients and
allocation kernels but excluding Python dispatch/JIT compilation time. The
cuTile API uses a fixed seed; PyTorch uses its own RNG and saves a bool mask.
Each backend is validated against the mask from its own forward on ones.
Bandwidth is effective gradient read + write traffic, excluding RNG/mask work.
Pass --large-matrices to benchmark (4096, 8192), (8192, 8192), and
(16384, 8192) inputs instead of the default vector sizes.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

ALL_BACKENDS = [
    ("cutile", "cuTile", ("orange", "-")) if is_backend_available("cutile") else None,
    ("torch", "PyTorch", ("green", "-")),
]
RESULTS = []
SIZES = [2**10, 2**14, 2**20, 2**22, 2**24]
LARGE_MATRIX_SIZES = [2**25, 2**26, 2**27]
MATRIX_COLS = 8192


def reference_dropout(x, seed, p=0.5, training=True, inplace=False, **kwargs):
    # Native PyTorch RNG does not share cuTile's stateless seeded mask.
    return torch.nn.functional.dropout(x, p, training, inplace)


register_impl("dropout", "torch")(reference_dropout)


def get_supported_backends():
    return [entry for entry in ALL_BACKENDS if entry is not None]


def create_benchmark_config(dtype, mode, large_matrices=False):
    backends, names, styles = zip(*get_supported_backends())
    prefix = "dropout-matrix" if large_matrices else "dropout"
    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=LARGE_MATRIX_SIZES if large_matrices else SIZES,
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="Effective GB/s",
        plot_name=f"{prefix}-{mode}-{str(dtype).split('.')[-1]}-GBps",
        args={"dtype": dtype, "mode": mode, "matrix_cols": MATRIX_COLS if large_matrices else 0},
    )


def measure_cuda_graph(run, stream):
    """Keep autograd's forward stream and the capture stream identical."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(32):
            run()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        events = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            events.append((start, end))
        stream.synchronize()
        samples.append(statistics.median(start.elapsed_time(end) / 32 for start, end in events))
    return samples


@triton.testing.perf_report(
    [
        create_benchmark_config(dtype, mode)
        for dtype in [torch.float16, torch.bfloat16, torch.float32]
        for mode in ["backward", "forward-backward"]
    ]
)
def bench_dropout_backward(N, backend, dtype, mode, device="cuda", matrix_cols=0):
    # Backward executes on the stream recorded by forward. In particular, an
    # already-created autograd graph on the legacy stream cannot be captured
    # on the fresh stream used internally by do_bench_cudagraph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        return _benchmark_on_stream(N, backend, dtype, mode, device, stream, matrix_cols)


def _benchmark_on_stream(N, backend, dtype, mode, device, stream, matrix_cols=0):
    seed, p = 11, 0.5
    if matrix_cols and N % matrix_cols:
        raise ValueError("N must be divisible by matrix_cols")
    shape = (N // matrix_cols, matrix_cols) if matrix_cols else (N,)
    x = torch.ones(shape, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    saved_bytes = 0

    def pack(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        y = tilegym.ops.dropout(x, seed, p=p, backend=backend)
    mask = y.detach() != 0
    dx = torch.autograd.grad(y, x, dy, retain_graph=True)[0]
    torch.testing.assert_close(dx, torch.where(mask, dy * 2, 0), rtol=0, atol=0)
    del mask, dx

    if mode == "backward":

        def run():
            return torch.autograd.grad(y, x, dy, retain_graph=True)[0]

    else:

        def run():
            output = tilegym.ops.dropout(x, seed, p=p, backend=backend)
            return torch.autograd.grad(output, x, dy)[0]

    run()  # Warm both forward and backward before timing or memory measurement.
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline
    samples_ms = measure_cuda_graph(run, stream)
    ms = statistics.median(samples_ms)
    effective_bytes = (2 if mode == "backward" else 4) * N * x.element_size()
    RESULTS.append(
        {
            "N": N,
            "shape": list(shape),
            "backend": backend,
            "dtype": str(dtype).split(".")[-1],
            "mode": mode,
            "median_us": ms * 1000,
            "samples_us": [sample * 1000 for sample in samples_ms],
            "effective_GBps": effective_bytes / (ms * 1e6),
            "forward_saved_tensor_bytes": saved_bytes,
            "incremental_peak_bytes": peak_extra,
        }
    )
    return effective_bytes / (ms * 1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tilegym-dropout-benchmark"))
    parser.add_argument(
        "--large-matrices", action="store_true", help="Benchmark 2D matrices with 2**25 to 2**27 elements"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.large_matrices:
        bench_dropout_backward.benchmarks = [
            create_benchmark_config(dtype, mode, large_matrices=True)
            for dtype in [torch.float16, torch.bfloat16, torch.float32]
            for mode in ["backward", "forward-backward"]
        ]
    bench_dropout_backward.run(print_data=True, save_path=str(args.output))
    import cuda.tile as ct

    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutile": ct.__version__,
        "triton": triton.__version__,
        "seed": 11,
        "p": 0.5,
        "large_matrices": args.large_matrices,
        "method": "median of 3 warmed CUDA Graph trials, 10 replays of 32 ops each; contiguous dy; no cache flush",
    }
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "results": RESULTS}, indent=2) + "\n")
