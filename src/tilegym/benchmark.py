# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Cold-cache CUDA Graph timing shared by benchmarks and autotuners."""

import ctypes
import math
import os
import re
import time
from contextlib import ExitStack
from contextlib import nullcontext

import torch

GRAPH_PROTOCOL = "cuda-graph-cold-span-v4"
EAGER_PROTOCOL = "cuda-eager-cold-span-v2"
FLUSH_BYTES = 256_000_000
GRAPH_REPEATS = 8
# Power capping lowers clocks after about 40 ms of continuous load on B200, and a
# saturated GPU needs about 100 ms idle to recover.
POWER_BURST_MS = 30.0
POWER_REST_MS = 100.0

TELEMETRY_ENV = "TILEGYM_SAMPLE_TELEMETRY"
TELEMETRY_MAX_READS = 64
# Per-sample reads are issued while the sample's GPU work is in flight; below this
# sample length they could outlast it and add host idle between samples.
TELEMETRY_MIN_SAMPLE_MS = 1.0
_NVML_CLOCK_SM = 1
_NVML_REASON_IDLE = 0x1
# Clocks-event reasons that are slowdowns: SW power cap, HW slowdown, SW thermal,
# HW thermal, HW power brake. Idle, applications-clocks and display-clock settings
# are not.
_NVML_REASON_LIMITERS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
_NVML_FI_POWER_INSTANT = 186
_NVML_FI_POWER_CURRENT_LIMIT = 190
_NVML_FI_POWER_REQUESTED_LIMIT = 192
_NVML_FI_EVENT_COUNTERS = {
    "sw_power_cap": 74,
    "sw_thermal": 269,
    "hw_thermal": 270,
    "hw_power_brake": 271,
}


class _NvmlFieldValue(ctypes.Structure):
    _fields_ = [
        ("fieldId", ctypes.c_uint),
        ("scopeId", ctypes.c_uint),
        ("timestamp", ctypes.c_longlong),
        ("latencyUsec", ctypes.c_longlong),
        ("valueType", ctypes.c_int),
        ("nvmlReturn", ctypes.c_int),
        ("value", ctypes.c_ulonglong),
    ]


class _DeviceTelemetry:
    """SM clock, instantaneous board power and clocks-event reasons of the current device, via NVML.

    Each per-sample read happens on the host while the sample's GPU work is in
    flight, so it never appears on the device timeline; samples shorter than
    TELEMETRY_MIN_SAMPLE_MS get no per-sample reads (the read could outlast the
    work and add host idle between samples), only the per-measurement block. The cumulative
    clocks-event counters (nanoseconds spent under each limiter) are read only
    at measurement boundaries because each read costs about half a
    millisecond. The library is loaded directly so no Python package is
    required; a missing library, an NVML error, or TILEGYM_SAMPLE_TELEMETRY=0
    leaves the telemetry fields unavailable instead of failing the measurement.
    """

    _lib = None
    _load_failed = False

    def __init__(self, enabled=True):
        self.handle = None
        self.reasons_fn = None
        self.status = "unavailable"
        if not enabled or os.environ.get(TELEMETRY_ENV, "1") == "0":
            self.status = "disabled"
            return
        lib = self._library()
        if lib is None:
            return
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        handle = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode(), ctypes.byref(handle)) != 0:
            return
        self.handle = handle
        self.reasons_fn = getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or getattr(
            lib, "nvmlDeviceGetCurrentClocksThrottleReasons", None
        )
        self.status = "recorded"

    @classmethod
    def _library(cls):
        if cls._lib is None and not cls._load_failed:
            try:
                lib = ctypes.CDLL("libnvidia-ml.so.1")
                if lib.nvmlInit_v2() != 0:
                    raise OSError("nvmlInit_v2 failed")
                cls._lib = lib
            except (OSError, AttributeError):
                cls._load_failed = True
        return cls._lib

    def _fields(self, ids):
        values = (_NvmlFieldValue * len(ids))()
        for i, field_id in enumerate(ids):
            values[i].fieldId = field_id
            values[i].scopeId = 0
        if self._lib.nvmlDeviceGetFieldValues(self.handle, len(ids), values) != 0:
            return {}
        out = {}
        for v in values:
            if v.nvmlReturn == 0:
                out[v.fieldId] = v.value & 0xFFFFFFFF if v.valueType == 1 else v.value
        return out

    def read(self):
        if self.handle is None:
            return None
        lib = self._lib
        clock = ctypes.c_uint()
        reasons = ctypes.c_ulonglong()
        record = {}
        if lib.nvmlDeviceGetClockInfo(self.handle, _NVML_CLOCK_SM, ctypes.byref(clock)) == 0:
            record["sm_clock_mhz"] = clock.value
        power = self._fields([_NVML_FI_POWER_INSTANT]).get(_NVML_FI_POWER_INSTANT)
        if power is not None:
            record["power_w"] = power / 1000
        if self.reasons_fn is not None and self.reasons_fn(self.handle, ctypes.byref(reasons)) == 0:
            record["clocks_event_reasons"] = reasons.value
        return record

    def counters(self):
        """Cumulative nanoseconds under each clock limiter, plus a host timestamp in ns."""
        if self.handle is None:
            return None
        values = self._fields(list(_NVML_FI_EVENT_COUNTERS.values()))
        return {
            "host_ns": time.perf_counter_ns(),
            **{name: values[fid] for name, fid in _NVML_FI_EVENT_COUNTERS.items() if fid in values},
        }

    def limits(self):
        """Power limits in watts and the maximum SM clock, read once per measurement."""
        if self.handle is None:
            return {}
        out = {}
        limit = ctypes.c_uint()
        if self._lib.nvmlDeviceGetEnforcedPowerLimit(self.handle, ctypes.byref(limit)) == 0:
            out["enforced_power_limit_w"] = limit.value / 1000
        fields = self._fields([_NVML_FI_POWER_REQUESTED_LIMIT, _NVML_FI_POWER_CURRENT_LIMIT])
        if _NVML_FI_POWER_REQUESTED_LIMIT in fields:
            out["requested_power_limit_w"] = fields[_NVML_FI_POWER_REQUESTED_LIMIT] / 1000
        if _NVML_FI_POWER_CURRENT_LIMIT in fields:
            out["current_power_limit_w"] = fields[_NVML_FI_POWER_CURRENT_LIMIT] / 1000
        if self._lib.nvmlDeviceGetMaxClockInfo(self.handle, _NVML_CLOCK_SM, ctypes.byref(limit)) == 0:
            out["max_sm_clock_mhz"] = limit.value
        return out


def _telemetry_stride(count):
    return max(1, -(-count // TELEMETRY_MAX_READS))


def _attach_telemetry(samples, readings, status, block=None):
    for sample in samples:
        sample["telemetry_status"] = status
    for index, record in readings:
        if record:
            samples[index]["telemetry"] = record
    if samples and block:
        samples[0]["telemetry_block"] = block
    return samples


def _telemetry_block(telemetry, start, end):
    """Limiter time fractions over one measurement from cumulative counters, plus the limits.

    The fractions cover the wall time of the timed loop, host gaps included, and the
    driver updates the counters and the clocks-event reasons on its own cadence
    (about 100 ms), so they describe the device state around the samples, not the
    state during one sample; sample durations remain the measurement.
    """
    block = telemetry.limits()
    if start and end and end["host_ns"] > start["host_ns"]:
        wall = end["host_ns"] - start["host_ns"]
        block["limiter_time_fraction"] = {
            name: min(1.0, max(0.0, (end[name] - start[name]) / wall))
            for name in _NVML_FI_EVENT_COUNTERS
            if name in start and name in end
        }
        block["block_wall_ms"] = wall / 1e6
    return block


def _telemetry_summary(samples):
    status = next((s["telemetry_status"] for s in samples if "telemetry_status" in s), "unavailable")
    summary = {"telemetry_status": status}
    if samples and samples[0].get("telemetry_block"):
        summary["telemetry_block"] = samples[0]["telemetry_block"]
    readings = [(index, s["telemetry"]) for index, s in enumerate(samples) if s.get("telemetry")]
    if not readings:
        return summary
    summary["telemetry_samples"] = [
        {
            "sample": index,
            **{k: (f"{v:#x}" if k == "clocks_event_reasons" else v) for k, v in record.items()},
        }
        for index, record in readings
    ]
    clocks = sorted(r["sm_clock_mhz"] for _, r in readings if "sm_clock_mhz" in r)
    if clocks:
        summary["sm_clock_mhz_min"] = clocks[0]
        summary["sm_clock_mhz_median"] = clocks[len(clocks) // 2]
        summary["sm_clock_mhz_max"] = clocks[-1]
    reasons = [r["clocks_event_reasons"] for _, r in readings if "clocks_event_reasons" in r]
    if reasons:
        summary["throttled_sample_fraction"] = sum(bool(r & _NVML_REASON_LIMITERS) for r in reasons) / len(reasons)
    return summary


def _available_graph_memory():
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    fraction = torch.cuda.get_per_process_memory_fraction()
    if fraction < 1:
        allocator_free = max(0, int(total_bytes * fraction) - torch.cuda.memory_reserved())
        free_bytes = min(free_bytes, allocator_free)
    return free_bytes


def iteration_counts(estimate_ms, warmup, rep, min_rep, max_rep):
    if not math.isfinite(estimate_ms) or estimate_ms <= 0:
        raise ValueError(f"Invalid GPU duration: {estimate_ms}")
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(min_rep, int(rep / estimate_ms))
    if max_rep > 0:
        n_repeat = min(n_repeat, max_rep)
        n_warmup = min(n_warmup, max(1, int(max_rep * warmup / rep)))
    return n_warmup, n_repeat


class _PowerPacer:
    """Rests the GPU before continuous work would outlast POWER_BURST_MS."""

    def __init__(self):
        self.busy_ms = 0.0
        self.rests = 0

    def admit(self, duration_ms):
        if self.busy_ms and self.busy_ms + duration_ms > POWER_BURST_MS:
            torch.cuda.synchronize()
            time.sleep(POWER_REST_MS / 1000)
            self.busy_ms = 0.0
            self.rests += 1
        self.busy_ms += duration_ms


class _CapturedCall:
    def __init__(self, fn, setup_fn, grad_to_none, fast_flush, *, cache=None, warmup=True, input_context=nullcontext):
        self.graph = torch.cuda.CUDAGraph()
        self.setup_graph = None
        self.input_context = input_context
        self.begin = torch.cuda.Event(enable_timing=True, external=True)
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        dtype = torch.int32 if fast_flush else torch.int8
        self.cache = (
            cache if cache is not None else torch.empty(FLUSH_BYTES // dtype.itemsize, dtype=dtype, device="cuda")
        )
        cache = None
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
        except BaseException:
            self.close()
            raise
        finally:
            caller.wait_stream(capture_stream)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.graph.reset()
        if self.setup_graph is not None:
            self.setup_graph.reset()
        self.cache = None

    def prepare(self):
        if self.setup_graph is not None:
            self.setup_graph.replay()
        self.cache.zero_()

    def sample(self):
        """Return the timed replay and the preparation span before it."""
        with self.input_context():
            self.begin.record()
            self.prepare()
            self.start.record()
            self.graph.replay()
            self.end.record()
            self.end.synchronize()
        return self.start.elapsed_time(self.end), self.begin.elapsed_time(self.start)

    def profile_samples(self, count, calls=None, telemetry=True, pacer=None, sample_ms=0.0, replay_ms=None):
        calls = [self] if calls is None else calls
        telemetry = _DeviceTelemetry(enabled=telemetry)
        stride = _telemetry_stride(count)
        # Gate on the replay alone: sample_ms also holds the preparation, which does
        # not overlap the read.
        replay_ms = sample_ms if replay_ms is None else replay_ms
        per_sample = telemetry.handle is not None and replay_ms >= TELEMETRY_MIN_SAMPLE_MS
        readings = []
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            # Boundary counters bracket the sample loop alone, so the limiter fractions
            # describe the timed work and its rests, not profiler start-up or parsing.
            counters_start = telemetry.counters()
            for index in range(count):
                call = calls[index % len(calls)]
                if pacer is not None:
                    pacer.admit(sample_ms)
                with call.input_context():
                    call.prepare()
                    torch.cuda.synchronize()
                    with torch.profiler.record_function("tilegym.graph.sample"):
                        call.graph.replay()
                        if per_sample and index % stride == 0:
                            readings.append((index, telemetry.read()))
                        torch.cuda.current_stream().synchronize()
            counters_end = telemetry.counters()
        samples = _profile_graph_samples(prof.profiler.kineto_results.events(), count, len(calls))
        block = _telemetry_block(telemetry, counters_start, counters_end) if telemetry.handle else None
        return _attach_telemetry(samples, readings, telemetry.status, block)


_GRAPH_LAUNCHES = ("cudaGraphLaunch", "cudaGraphLaunch_ptsz", "cuGraphLaunch", "cuGraphLaunch_ptsz")


def _profile_samples(events, count, scope_name, is_launch, kind):
    scopes, launches = [], []
    device_events = {}
    names = {}
    for event in events:
        if event.is_hidden_event():
            continue
        device = event.device_type()
        if device == torch.autograd.DeviceType.CPU:
            if event.name() == scope_name:
                scopes.append((event.start_ns(), event.end_ns()))
            elif is_launch(event.name()):
                launches.append((event.start_ns(), event.end_ns(), event.correlation_id()))
        elif device == torch.autograd.DeviceType.CUDA and not event.is_user_annotation():
            name = event.name()
            if name not in names:
                names[name] = torch._C._demangle(name)
            device_events.setdefault(event.correlation_id(), []).append((event.start_ns(), event.end_ns(), names[name]))
    scopes.sort()
    launches.sort()
    if len(scopes) != count:
        raise RuntimeError(f"Expected {count} profiler scopes, found {len(scopes)}")
    samples = []
    index = 0
    for scope_start, scope_end in scopes:
        while index < len(launches) and launches[index][0] < scope_start:
            index += 1
        selected = []
        correlations = set()
        while index < len(launches) and launches[index][0] < scope_end:
            launch_start, launch_end, correlation = launches[index]
            if launch_end > scope_end:
                raise RuntimeError(f"{kind} launch crosses a synchronized sample boundary")
            if correlation not in correlations:
                correlations.add(correlation)
                selected.extend(device_events.get(correlation, []))
            index += 1
        if not selected:
            raise RuntimeError(f"{kind} sample produced no profiled GPU activities")
        selected.sort()
        samples.append(
            {
                "span_ms": (max(event[1] for event in selected) - selected[0][0]) / 1_000_000,
                "activities": [{"name": name, "duration_us": (end - start) / 1000} for start, end, name in selected],
            }
        )
    return samples


def _profile_graph_samples(events, count, graph_count):
    samples = _profile_samples(events, count, "tilegym.graph.sample", _GRAPH_LAUNCHES.__contains__, "CUDA Graph")
    for index, sample in enumerate(samples):
        sample["graph_index"] = index % graph_count
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


def _check_budget(warmup, rep, min_rep, initial_rep, max_rep):
    if warmup < 0 or rep <= 0 or min_rep < 1 or initial_rep < 1 or max_rep < 0:
        raise ValueError("Invalid benchmark iteration budget")
    if max_rep and max_rep < min_rep:
        raise ValueError("max_rep must be zero or at least min_rep")


def _summarize(profiled, kernel_filter, protocol, collect_kernel_times, kind):
    patterns = None
    if kernel_filter is not None:
        patterns = [re.compile(p) for p in ([kernel_filter] if isinstance(kernel_filter, str) else kernel_filter)]
    samples = []
    for sample in profiled:
        if patterns is None:
            samples.append(sample["span_ms"])
        else:
            matched = [k for k in sample["activities"] if any(p.search(k["name"]) for p in patterns)]
            if not matched:
                raise RuntimeError(f"No {kind} kernels matched {kernel_filter!r}")
            samples.append(sum(k["duration_us"] for k in matched) / 1000)
    times = torch.tensor(samples, dtype=torch.float64)
    mean = times.mean().item()
    std = times.std(correction=0).item()
    result = {
        "mean": mean,
        "std": std,
        "rel_std": 100 * std / mean if mean else 0,
        "median": times.median().item(),
        "min": times.min().item(),
        "max": times.max().item(),
        "nrep": len(samples),
        "peak_mem_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
        "measurement_protocol": protocol + ("-filtered" if patterns is not None else ""),
        "cache_flush_bytes": FLUSH_BYTES,
        "samples_ms": samples,
    }
    result.update(_telemetry_summary(profiled))
    if collect_kernel_times or patterns is not None:
        kernel_times = _sample_kernel_records(profiled)
        result["kernel_times"] = kernel_times
        result["kernel_metadata_status"] = "recorded" if kernel_times else "unavailable"
        result["kernel_metadata_source"] = "timed_samples"
    return result, times


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
    telemetry=True,
    pacer=None,
):
    """Measure cold-cache device spans with balanced samples across graphs.

    Calibration uses at most initial_rep samples and stops once their combined
    GPU duration reaches the larger of the warmup and measurement budgets.
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
    Calibration, warmup, and timed replays run in bursts of at most POWER_BURST_MS
    of estimated GPU time, preparation included, separated by POWER_REST_MS idle,
    so power capping does not lower clocks during timing. A replay longer than
    POWER_BURST_MS still runs whole after a rest, so its time includes any capping
    the kernel itself triggers. A caller timing several functions in a row passes
    one pacer so that bursts continue across calls.
    telemetry=False skips the NVML reads, for callers such as autotuning that only
    rank candidates and would otherwise pay the reads per candidate.
    """
    _check_budget(warmup, rep, min_rep, initial_rep, max_rep)
    if graph_repeats < 1:
        raise ValueError("Invalid benchmark iteration budget")
    with ExitStack() as captures:
        call = captures.enter_context(
            _CapturedCall(fn, setup_fn, grad_to_none, fast_flush, input_context=input_context)
        )
        graph_pool = call.graph.pool()
        graph_pool_reserved_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment.get("segment_pool_id") == graph_pool
        )
        pacer = pacer if pacer is not None else _PowerPacer()
        initial_rests = pacer.rests
        # The first preparation also pays one-time host work such as lazy module
        # loading, and any span can include host stalls that leave the GPU idle,
        # so the shortest span after the first is closest to its GPU work. A
        # single calibration sample already fills the budget, so it has none.
        calibration_ms, prepare_ms = 0.0, []
        for calibration_count in range(1, initial_rep + 1):
            pacer.admit(calibration_ms / max(1, calibration_count - 1) + min(prepare_ms[1:], default=0.0))
            timed_ms, span_ms = call.sample()
            calibration_ms += timed_ms
            prepare_ms.append(span_ms)
            if calibration_ms >= max(warmup, rep):
                break
        estimate = calibration_ms / calibration_count
        sample_ms = estimate + min(prepare_ms[1:], default=0.0)
        n_warmup, n_repeat = iteration_counts(estimate, warmup, rep, min_rep, max_rep)
        n_graphs = min(graph_repeats, n_repeat)
        requested_graph_count = n_graphs
        if graph_pool_reserved_bytes:
            n_graphs = min(n_graphs, 1 + _available_graph_memory() // graph_pool_reserved_bytes)
        graph_memory_limited = n_graphs < requested_graph_count
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
                captures.enter_context(
                    _CapturedCall(fn, setup_fn, grad_to_none, fast_flush, cache=call.cache, input_context=input_context)
                )
            )
        for _ in range(math.ceil(n_warmup / n_graphs)):
            for captured in calls:
                pacer.admit(sample_ms)
                with captured.input_context():
                    captured.prepare()
                    captured.graph.replay()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        profiled = call.profile_samples(
            n_repeat, calls=calls, telemetry=telemetry, pacer=pacer, sample_ms=sample_ms, replay_ms=estimate
        )
        result, times = _summarize(profiled, kernel_filter, GRAPH_PROTOCOL, collect_kernel_times, "CUDA Graph")
        graph_means = torch.stack([times[index::n_graphs].mean() for index in range(n_graphs)])
        graph_std = graph_means.std(correction=0).item()
        mean = result["mean"]
        result.update(
            {
                "graph_count": n_graphs,
                "requested_graph_count": requested_graph_count,
                "graph_memory_limited": graph_memory_limited,
                "graph_pool_reserved_bytes": graph_pool_reserved_bytes,
                "samples_per_graph": per_graph,
                "graph_means_ms": graph_means.tolist(),
                "graph_mean_std_ms": graph_std,
                "graph_mean_rel_std": 100 * graph_std / mean if mean else 0,
                "power_rests": pacer.rests - initial_rests,
            }
        )
        return result


def benchmark_eager(
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
):
    """Measure cold-cache device spans of direct launches, for work that cannot be captured.

    Samples follow benchmark_cuda_graph: setup and the cache flush run before
    each sample, a sample spans its first to last GPU activity, including the
    host-staged copies its launches issue, and work is paced the same way.
    """
    _check_budget(warmup, rep, min_rep, initial_rep, max_rep)
    dtype = torch.int32 if fast_flush else torch.int8
    cache = torch.empty(FLUSH_BYTES // dtype.itemsize, dtype=dtype, device="cuda")
    begin = torch.cuda.Event(enable_timing=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def prepare():
        if setup_fn is not None:
            setup_fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        cache.zero_()

    def sample():
        begin.record()
        prepare()
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end), begin.elapsed_time(start)

    pacer = _PowerPacer()
    pacer.admit(sample()[0])
    calibration_ms, prepare_ms = 0.0, []
    for calibration_count in range(1, initial_rep + 1):
        pacer.admit(calibration_ms / max(1, calibration_count - 1) + min(prepare_ms[1:], default=0.0))
        timed_ms, span_ms = sample()
        calibration_ms += timed_ms
        prepare_ms.append(span_ms)
        if calibration_ms >= max(warmup, rep):
            break
    estimate = calibration_ms / calibration_count
    sample_ms = estimate + min(prepare_ms[1:], default=0.0)
    n_warmup, n_repeat = iteration_counts(estimate, warmup, rep, min_rep, max_rep)
    for _ in range(n_warmup):
        pacer.admit(sample_ms)
        prepare()
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    telemetry = _DeviceTelemetry()
    stride = _telemetry_stride(n_repeat)
    per_sample = telemetry.handle is not None and estimate >= TELEMETRY_MIN_SAMPLE_MS
    readings = []
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        counters_start = telemetry.counters()
        for index in range(n_repeat):
            pacer.admit(sample_ms)
            prepare()
            torch.cuda.synchronize()
            with torch.profiler.record_function("tilegym.eager.sample"):
                fn()
                if per_sample and index % stride == 0:
                    readings.append((index, telemetry.read()))
                torch.cuda.synchronize()
        counters_end = telemetry.counters()
    # Only CUDA runtime and driver calls share correlation IDs with device activities.
    profiled = _profile_samples(
        prof.profiler.kineto_results.events(),
        n_repeat,
        "tilegym.eager.sample",
        lambda name: name.startswith("cu"),
        "Eager",
    )
    block = _telemetry_block(telemetry, counters_start, counters_end) if telemetry.handle else None
    _attach_telemetry(profiled, readings, telemetry.status, block)
    result = _summarize(profiled, kernel_filter, EAGER_PROTOCOL, collect_kernel_times, "eager")[0]
    result["power_rests"] = pacer.rests
    return result


def autotune_cutile_cuda_graph(stream, grid, kernel, args):
    import cuda.tile as ct

    # Upstream owns warmup; each callback must launch once on its prepared inputs.
    with torch.cuda.stream(stream):
        with _CapturedCall(
            lambda: ct.launch(torch.cuda.current_stream(), grid, kernel, args), None, None, True, warmup=False
        ) as call:
            return call.profile_samples(1, telemetry=False)[0]["span_ms"] * 1000
