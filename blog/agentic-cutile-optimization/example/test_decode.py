# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch
from decode_cutile import paged_decode_attention
from decode_cutile import reference_decode

# (B, H, H_kv, D, PAGE, seq_lens)
CASES = [
    (1, 8, 8, 128, 16, [16]),
    (2, 8, 2, 128, 16, [100, 37]),
    (4, 32, 8, 128, 16, [1024, 1, 513, 4096]),
    (3, 16, 4, 64, 32, [2000, 64, 999]),
    (8, 32, 8, 128, 64, [8192] * 8),
]


def make(B, H, H_kv, D, PAGE, seq_lens):
    torch.manual_seed(0)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    max_pages = math.ceil(max(seq_lens) / PAGE)
    num_pages = B * max_pages + 8
    k = torch.randn(num_pages, PAGE, H_kv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(num_pages, PAGE, H_kv, D, device="cuda", dtype=torch.float16)
    bt = torch.randperm(num_pages, device="cuda")[: B * max_pages].reshape(B, max_pages).to(torch.int32)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    return q, k, v, bt, seq


@pytest.mark.parametrize("B,H,H_kv,D,PAGE,seq_lens", CASES)
def test_decode_matches_reference(B, H, H_kv, D, PAGE, seq_lens):
    q, k, v, bt, seq = make(B, H, H_kv, D, PAGE, seq_lens)
    out = paged_decode_attention(q, k, v, bt, seq)
    ref = reference_decode(q, k, v, bt, seq)
    assert out.shape == q.shape and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
