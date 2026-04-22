# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile


def profile_with_l2flush(fn, warmup=20, rep=100):
    """Measure kernel-only GPU time using torch.profiler, with L2 flush between reps."""
    l2_size = torch.cuda.get_device_properties(0).L2_cache_size
    l2_flush = torch.empty(l2_size // 4, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(rep):
        l2_flush.zero_()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kernel_us = sum(evt.device_time_total for evt in prof.key_averages() if evt.device_time_total > 0)
        times_us.append(kernel_us)

    times_us.sort()
    median_us = times_us[len(times_us) // 2]
    return median_us / 1000.0  # ms
