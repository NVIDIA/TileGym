# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import datetime
import os
import zipfile

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function

from tilegym_hf_bench.kernel_filters import KernelFilter


def run_torch_profiler(forward_wrapper, args, case_id, avg_time, summary_line):
    print("Profile the model...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
        record_shapes=False,
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = forward_wrapper.forward()

    with torch.no_grad():
        torch.cuda.cudart().cudaProfilerStart()
        _ = forward_wrapper.forward()
        torch.cuda.cudart().cudaProfilerStop()

    filtered_results = []
    kernel_filter = KernelFilter()

    for item in prof.key_averages():
        if kernel_filter.contains(item.key):
            filtered_results.append(item)

    if filtered_results:
        print("\n===== FILTERED PROFILER RESULTS =====")
        headers = [
            "Name",
            "CPU time total (us)",
            "CPU time avg (us)",
            "CUDA time total (us)",
            "CUDA time avg (us)",
            "Count",
        ]
        row_format = "{:<80} {:<20} {:<20} {:<20} {:<20} {:<10}"

        print(row_format.format(*headers))
        print("-" * 140)
        total_device_time = 0.0
        for item in filtered_results:
            if kernel_filter.contains(item.key):
                total_device_time += item.device_time_total
            print(
                row_format.format(
                    item.key[:50],
                    f"{item.cpu_time_total:.2f}",
                    f"{item.cpu_time:.2f}",
                    f"{item.device_time_total:.2f}",
                    f"{item.device_time:.2f}",
                    item.count,
                )
            )
        total_device_time_pct = 0.0 if avg_time <= 0 else (total_device_time / 1_000_000.0) / avg_time * 100.0
        print(
            f"Total device time: {total_device_time:.2f} us",
            f"Total device time %: {total_device_time_pct:.2f}%",
        )
    print(prof.key_averages().table(row_limit=1))

    all_results = prof.key_averages()
    all_results = sorted(all_results, key=lambda x: x.device_time_total, reverse=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{log_dir}/profiler_results_{case_id}_{timestamp}.csv"

    with open(filename, "w") as f:
        f.write(
            "Name,CPU_time_total_us,CPU_time_avg_us,CUDA_time_total_us,CUDA_time_avg_us,Self_CPU_time_total_us,Self_CUDA_time_total_us,Count,Filtered,DeviceType\n"
        )

        for item in all_results:
            device_type_str = str(item.device_type) if hasattr(item, "device_type") else ""
            f.write(
                f'"{item.key}",{item.cpu_time_total:.2f},{item.cpu_time:.2f},{item.device_time_total:.2f},{item.device_time:.2f},{item.self_cpu_time_total:.2f},{item.self_device_time_total:.2f},{item.count},{kernel_filter.contains(item.key)},{device_type_str}\n'
            )
    trace_filename = f"{log_dir}/trace_{case_id}_{timestamp}.json"
    prof.export_chrome_trace(trace_filename)

    zip_filename = f"{log_dir}/trace_{case_id}_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(trace_filename)
    os.remove(trace_filename)

    print(f"Trace file zipped to {zip_filename}")
    print(f"Profiler results saved to {filename}")
    print(prof.key_averages().table(row_limit=1))

    if args.summary_file:
        cuda_kernel_time_us = 0.0
        for item in all_results:
            device_type_str = str(item.device_type) if hasattr(item, "device_type") else ""
            if device_type_str == "DeviceType.CUDA" and item.key != "model_inference":
                cuda_kernel_time_us += item.self_device_time_total

        cuda_kernel_time_ms = cuda_kernel_time_us / 1000.0
        updated_summary_line = f"{summary_line} | {cuda_kernel_time_ms:>10.1f}\n"

        lines = []
        if os.path.exists(args.summary_file):
            with open(args.summary_file, encoding="utf-8") as f:
                lines = f.readlines()

        if lines:
            lines[-1] = updated_summary_line
        else:
            lines = [updated_summary_line]

        with open(args.summary_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"Updated summary with CUDA kernel time: {cuda_kernel_time_ms:.1f} ms")
