# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Additional interface/numerical checks; official tests stay unchanged."""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "example"))
from bench_decode import CASES
from bench_decode import H_KV
from bench_decode import PAGE
from bench_decode import D
from bench_decode import H
from decode_cutile import paged_decode_attention
from decode_cutile import reference_decode
from test_decode import make

for B, S in CASES:
    q, k, v, bt, seq = make(B, H, H_KV, D, PAGE, [S] * B)
    out = paged_decode_attention(q, k, v, bt, seq)
    kd = k[bt.long()].reshape(B, S, H_KV, D).transpose(1, 2).contiguous()
    vd = v[bt.long()].reshape(B, S, H_KV, D).transpose(1, 2).contiguous()
    ref = F.scaled_dot_product_attention(q[:, :, None, :], kd, vd, enable_gqa=True).squeeze(2)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print(f"benchmark B{B}_S{S} max_abs={(out - ref).abs().max().item():.6g} PASS", flush=True)
    del q, k, v, bt, seq, kd, vd, out, ref

cases = [
    (1, 8, 8, 128, 16, [1]),
    (4, 12, 4, 64, 16, [1, 17, 255, 1025]),
    (2, 24, 3, 128, 32, [1000, 13]),
    (2, 64, 2, 64, 64, [4097, 1]),
    (3, 16, 4, 128, 16, [63, 8000, 257]),
    (2, 8, 2, 128, 256, [511, 257]),
    (2, 8, 2, 64, 512, [1001, 1]),
]
for case in cases:
    for scale in [None, 0.0, -0.25, 2.0]:
        q, k, v, bt, seq = make(*case)
        # Noncontiguous query, KV, block table, and sequence arrays.
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)
        bt = bt.transpose(0, 1).contiguous().transpose(0, 1)
        seq = torch.stack((seq, seq), dim=1)[:, 0]
        out = paged_decode_attention(q, k, v, bt, seq, scale=scale)
        ref = reference_decode(q, k, v, bt, seq, scale=scale)
        assert out.shape == q.shape and out.dtype == q.dtype
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        print(f"extended {case} scale={scale} max_abs={(out - ref).abs().max().item():.6g} PASS", flush=True)
print("ALL EXTENDED CHECKS PASS", flush=True)
