# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Time decode_cutile.paged_decode_attention against PyTorch SDPA on the same KV (gathered once, dense).

Cases: H=32 query heads, H_kv=8, D=128, page 16, equal seq_lens per batch; batch and KV length swept.
Usage: python bench_decode.py [--json out.json]
Timing: CUDA events, L2 flushed between iterations, median of 30 after 10 warmups.
JSON rows: {"case", "cutile_ms", "ref_ms"}.
"""

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F
from decode_cutile import paged_decode_attention

CASES = [(1, 4096), (1, 32768), (8, 4096), (8, 32768), (64, 2048), (64, 8192)]  # (B, S)
H, H_KV, D, PAGE = 32, 8, 128, 16


def time_ms(fn, flush, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    rows = []
    print(f"{'B':>3} {'S':>6} {'cuTile ms':>10} {'SDPA ms':>9} {'Speedup vs SDPA':>15} {'KV GB/s cuTile':>15}")
    for B, S in CASES:
        torch.manual_seed(0)
        max_pages = S // PAGE
        num_pages = B * max_pages
        k = torch.randn(num_pages, PAGE, H_KV, D, device="cuda", dtype=torch.float16)
        v = torch.randn(num_pages, PAGE, H_KV, D, device="cuda", dtype=torch.float16)
        bt = torch.randperm(num_pages, device="cuda").reshape(B, max_pages).to(torch.int32)
        seq = torch.full((B,), S, dtype=torch.int32, device="cuda")
        q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        # dense reference inputs (gathered once, outside the timed region)
        kd = k[bt.long()].reshape(B, S, H_KV, D).transpose(1, 2).contiguous()  # (B,Hkv,S,D)
        vd = v[bt.long()].reshape(B, S, H_KV, D).transpose(1, 2).contiguous()
        qd = q.reshape(B, H, 1, D)
        ct_ms = time_ms(lambda: paged_decode_attention(q, k, v, bt, seq), flush)
        ref_ms = time_ms(lambda: F.scaled_dot_product_attention(qd, kd, vd, enable_gqa=True), flush)
        kv_gb = 2 * B * S * H_KV * D * 2 / 1e9
        print(f"{B:3d} {S:6d} {ct_ms:10.3f} {ref_ms:9.3f} {ref_ms / ct_ms:15.3f} {kv_gb / ct_ms * 1e3:15.0f}")
        rows.append({"case": f"B{B}_S{S}", "cutile_ms": ct_ms, "ref_ms": ref_ms})
        del k, v, kd, vd
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
