# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def addmm_f32_(acc, a, b):
    """acc += a @ b on an fp32 accumulator, accumulating inside the GEMM (beta=1)."""
    if a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float16):
        return torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
    return acc.addmm_(a.float(), b.float())  # .float() is a no-op for fp32


def mm_f32(a, b):
    """a @ b with an fp32 result; bf16/fp16 operands stay on tensor cores (fp32 accumulate and output)."""
    if a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float16):
        return torch.mm(a, b, out_dtype=torch.float32)
    return a.float() @ b.float()  # .float() is a no-op for fp32
