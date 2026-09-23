# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Cold-cache CUDA Graph timing shared by benchmarks and autotuners."""

import math
import re
from contextlib import nullcontext

import torch

GRAPH_PROTOCOL = "cuda-graph-cold-span-v3"
FLUSH_BYTES = 256_000_000
GRAPH_REPEATS = 8


def iteration_counts(estimate_ms, warmup, rep, min_rep, max_rep):
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        raise ValueError(f"Invalid GPU duration: {estimate_ms}")
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(min_rep, int(rep / estimate_ms))
    if max_rep > 0:
        n_repeat = min(n_repeat, max_rep)
        n_warmup = min(n_warmup, max(1, int(max_rep * warmup / rep)))
    return n_warmup, n_repeat


class _CapturedCall:
    def __init__(self, fn, setup_fn, grad_to_none, fast_flush, *, cache=None, warmup=True, input_context=nullcontext):
        self.graph = torch.cuda.CUDAGraph()
        self.setup_graph = None
        self.input_context = input_context
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        dtype = torch.int32 if fast_flush else torch.int8
        self.cache = (
            cache if cache is not None else torch.empty(FLUSH_BYTES // dtype.itemsize, dtype=dtype, device="cuda")
        )
        caller = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(caller)
        try:
            if warmup:
                with torch.cuda.stream(capture_stream), self.input_context():
                    if setup_fn is not None:
                        setup_fn()
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    fn()
            capture_stream.synchronize()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.stream(capture_stream), self.input_context():
                if setup_fn is not None:
                    self.setup_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.setup_graph, pool=pool, stream=capture_stream):
                        setup_fn()
                with torch.cuda.graph(self.graph, pool=pool, stream=capture_stream):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    fn()
        finally:
            caller.wait_stream(capture_stream)

    def prepare(self):
        if self.setup_graph is not None:
            self.setup_graph.replay()
        self.cache.zero_()

    def sample(self):
        with self.input_context():
            self.prepare()
            self.start.record()
            self.graph.replay()
            self.end.record()
            self.end.synchronize()
        return self.start.elapsed_time(self.end)

    def profile_samples(self, count, calls=None):
        calls = [self] if calls is None else calls
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for index in range(count):
                call = calls[index % len(calls)]
                with call.input_context():
                    call.prepare()
                    torch.cuda.synchronize()
                    with torch.profiler.record_function("tilegym.graph.sample"):
                        call.graph.replay()
                        torch.cuda.current_stream().synchronize()
        # Flat timestamps avoid FunctionEvent construction and CPU call-tree processing.
        scopes, device_events = [], []
        names = {}
        for event in prof.profiler.kineto_results.events():
            if event.is_hidden_event():
                continue
            device = event.device_type()
            if device == torch.autograd.DeviceType.CPU and event.name() == "tilegym.graph.sample":
                scopes.append((event.start_ns(), event.end_ns()))
            elif device == torch.autograd.DeviceType.CUDA and not event.is_user_annotation():
                name = event.name()
                if name not in names:
                    names[name] = torch._C._demangle(name)
                device_events.append((event.start_ns(), event.end_ns(), names[name]))
        scopes.sort(key=lambda event: event[0])
        device_events.sort(key=lambda event: event[0])
        if len(scopes) != count:
            raise RuntimeError(f"Expected {count} profiler scopes, found {len(scopes)}")
        samples = []
        index = 0
        for sample_index, (scope_start, scope_end) in enumerate(scopes):
            while index < len(device_events) and device_events[index][1] <= scope_start:
                index += 1
            selected = []
            while index < len(device_events) and device_events[index][0] < scope_end:
                event = device_events[index]
                if event[0] < scope_start or event[1] > scope_end:
                    raise RuntimeError("GPU activity crosses a synchronized sample boundary")
                selected.append(event)
                index += 1
            if not selected:
                raise RuntimeError("CUDA Graph replay produced no profiled GPU activities")
            samples.append(
                {
                    "graph_index": sample_index % len(calls),
                    "span_ms": (max(event[1] for event in selected) - selected[0][0]) / 1_000_000,
                    "activities": [
                        {"name": name, "duration_us": (end - start) / 1000} for start, end, name in selected
                    ],
                }
            )
        return samples


def _sample_kernel_records(samples):
    records = {}
    for sample in samples:
        for activity in sample["activities"]:
            name = activity["name"]
            record = records.setdefault(name, {"name": name, "duration_us": 0.0, "count": 0})
            record["duration_us"] += activity["duration_us"]
            record["count"] += 1
    return sorted(
        [
            {
                "name": record["name"],
                "self_time_us": record["duration_us"] / len(samples),
                "total_time_us": record["duration_us"] / len(samples),
                "count": record["count"] / len(samples),
                "sample_count": len(samples),
            }
            for record in records.values()
        ],
        key=lambda record: record["self_time_us"],
        reverse=True,
    )


def benchmark_cuda_graph(
    fn,
    warmup=100.0,
    rep=50.0,
    min_rep=2,
    initial_rep=5,
    grad_to_none=None,
    fast_flush=True,
    *,
    max_rep=1000,
    setup_fn=None,
    kernel_filter=None,
    collect_kernel_times=False,
    graph_repeats=GRAPH_REPEATS,
    input_context=nullcontext,
):
    """Measure cold-cache device spans with balanced samples across graphs.

    Compilation and tuning finish before capture. An optional setup graph runs
    before the flush and timed graph, including when recreating backward inputs.
    Capture errors propagate; this function never switches to eager execution.
    CUPTI timestamps retain copies and gaps between GPU operations, excluding
    timing-event overhead. A kernel filter explicitly sums matching activities.
    Optional kernel metadata is averaged over the same timed replays.
    Multiple independent captures include graph-instance and allocation effects
    that repeated replay of one graph cannot estimate. Every graph contributes
    the same number of samples; no graph or sample is selected for being faster.
    An optional input context wraps every warmup, capture, and replay. Its entry
    runs before the cache flush and its exit after timing, outside graph capture.
    """
    if warmup < 0 or rep <= 0 or min_rep < 1 or initial_rep < 1 or max_rep < 0 or graph_repeats < 1:
        raise ValueError("Invalid benchmark iteration budget")
    if max_rep and max_rep < min_rep:
        raise ValueError("max_rep must be zero or at least min_rep")
    call = _CapturedCall(fn, setup_fn, grad_to_none, fast_flush, input_context=input_context)
    estimate = sum(call.sample() for _ in range(initial_rep)) / initial_rep
    n_warmup, n_repeat = iteration_counts(estimate, warmup, rep, min_rep, max_rep)
    n_graphs = min(graph_repeats, n_repeat)
    while True:
        per_graph = math.ceil(n_repeat / n_graphs)
        if max_rep:
            per_graph = min(per_graph, max_rep // n_graphs)
        if per_graph * n_graphs >= min_rep:
            break
        n_graphs -= 1
    n_repeat = per_graph * n_graphs
    calls = [call]
    for _ in range(n_graphs - 1):
        calls.append(
            _CapturedCall(fn, setup_fn, grad_to_none, fast_flush, cache=call.cache, input_context=input_context)
        )
    for _ in range(math.ceil(n_warmup / n_graphs)):
        for captured in calls:
            with captured.input_context():
                captured.prepare()
                captured.graph.replay()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    patterns = None
    if kernel_filter is not None:
        patterns = [re.compile(p) for p in ([kernel_filter] if isinstance(kernel_filter, str) else kernel_filter)]
    profiled = call.profile_samples(n_repeat, calls=calls)
    samples = []
    kernel_times = []
    for sample in profiled:
        if patterns is None:
            samples.append(sample["span_ms"])
        else:
            matched = [k for k in sample["activities"] if any(p.search(k["name"]) for p in patterns)]
            if not matched:
                raise RuntimeError(f"No CUDA Graph kernels matched {kernel_filter!r}")
            samples.append(sum(k["duration_us"] for k in matched) / 1000)
    times = torch.tensor(samples, dtype=torch.float64)
    mean = times.mean().item()
    std = times.std(correction=0).item()
    graph_means = torch.stack([times[index::n_graphs].mean() for index in range(n_graphs)])
    graph_std = graph_means.std(correction=0).item()
    result = {
        "mean": mean,
        "std": std,
        "rel_std": 100 * std / mean if mean else 0,
        "median": times.median().item(),
        "min": times.min().item(),
        "max": times.max().item(),
        "nrep": n_repeat,
        "peak_mem_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "measurement_protocol": GRAPH_PROTOCOL + ("-filtered" if patterns is not None else ""),
        "cache_flush_bytes": FLUSH_BYTES,
        "samples_ms": samples,
        "graph_count": n_graphs,
        "samples_per_graph": per_graph,
        "graph_means_ms": graph_means.tolist(),
        "graph_mean_std_ms": graph_std,
        "graph_mean_rel_std": 100 * graph_std / mean if mean else 0,
    }
    if collect_kernel_times or patterns is not None:
        kernel_times = _sample_kernel_records(profiled)
        result["kernel_times"] = kernel_times
        result["kernel_metadata_status"] = "recorded" if kernel_times else "unavailable"
        result["kernel_metadata_source"] = "timed_samples"
    return result


def autotune_cutile_cuda_graph(stream, grid, kernel, args):
    import cuda.tile as ct

    # Upstream owns warmup; each callback must launch once on its prepared inputs.
    with torch.cuda.stream(stream):
        call = _CapturedCall(
            lambda: ct.launch(torch.cuda.current_stream(), grid, kernel, args), None, None, True, warmup=False
        )
        return call.profile_samples(1)[0]["span_ms"] * 1000
