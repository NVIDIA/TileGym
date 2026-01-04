# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
from tests import common


class Test_MLADecodingSplitKV(common.PyTestCase):
    @staticmethod
    def reference(q, qpe, kv, kpe, sm_scale=None, compute_dtype=torch.half):
        """Reference implementation using PyTorch for MLA decoding (same as test_mla_decoding.py)"""
        if sm_scale is None:
            sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

        qkv_dtype = q.dtype
        q = q.to(compute_dtype)
        qpe = qpe.to(compute_dtype)
        kv = kv.to(compute_dtype)
        kpe = kpe.to(compute_dtype)

        # Compute attention scores: Q*K^T + QPE*KPE^T
        qk = torch.matmul(q, kv.transpose(1, 2)).float()
        if kpe.numel() > 0:
            qk = qk + torch.matmul(qpe, kpe.transpose(1, 2)).float()

        qk = qk * sm_scale

        # Apply softmax
        m = torch.max(qk, dim=-1)[0]
        p = torch.exp(qk - m.unsqueeze(-1))
        l = torch.sum(p, dim=-1)
        p = p / (l.unsqueeze(-1))

        # Apply attention to values
        o = torch.matmul(p.to(qkv_dtype).to(compute_dtype), kv).to(qkv_dtype)
        return o

    @staticmethod
    def _get_sm_scale(q, qpe):
        """Calculate the default attention scale factor"""
        return 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    _backends = ["cutile"]

    @pytest.mark.parametrize("num_heads", [16, 32])
    @pytest.mark.parametrize("seq_len", [129, 1024, 8192, 11049])
    @pytest.mark.parametrize("kv_len_per_split", [128, 512])
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, num_heads, seq_len, kv_len_per_split, dtype, backend, arch):
        """Test functional correctness of MLA decoding with split-kv"""
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping MLA Split-KV test")

        # Test parameters
        batch_size = 1
        head_dim = 512
        kpe_dim = 64

        # Create random input tensors
        torch.manual_seed(42)  # For reproducibility
        device = torch.device("cuda")

        q = torch.randn(batch_size, num_heads, head_dim, device=device).to(dtype)
        qpe = torch.randn(batch_size, num_heads, kpe_dim, device=device).to(dtype)
        kv = torch.randn(batch_size, seq_len, head_dim, device=device).to(dtype)
        kpe = torch.randn(batch_size, seq_len, kpe_dim, device=device).to(dtype)

        # Compute softmax scale
        sm_scale = self._get_sm_scale(q, qpe)

        def split_kv_fn():
            return tilegym.ops.mla_decoding_split_kv(q, qpe, kv, kpe, sm_scale, kv_len_per_split)

        def ref_fn():
            return self.reference(q, qpe, kv, kpe, sm_scale)

        self.assertCorrectness(split_kv_fn, ref_fn, {}, atol=1e-2, rtol=1e-2, multiple_outputs=False)
