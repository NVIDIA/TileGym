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
    _perf_frameworks = _backends + ["pytorch"]

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

    @pytest.mark.parametrize("batch_size, num_heads, head_dim, kpe_dim", [(1, 16, 512, 64)])
    @pytest.mark.parametrize(
        "seq_len",
        [2**9, 2**10, 2**11, 2**12, 2**13] + [11049, 31079],
    )
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        kpe_dim,
        dtype,
        framework,
        record_property,
    ):
        """Performance test for MLA decoding with Split-KV"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability()[0] == 12:
            pytest.xfail(
                "Shared memory exhaustion on sm120: MLADecodingSplitKV requires 131,080 B > hardware limit 102,400 B"
            )

        self.setUp()
        device = torch.device("cuda")

        # Create test data with specified dtype
        torch.manual_seed(42)  # For reproducibility
        q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
        qpe = torch.randn(batch_size, num_heads, kpe_dim, device=device, dtype=dtype)
        kv = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=dtype)
        kpe = torch.randn(batch_size, seq_len, kpe_dim, device=device, dtype=dtype)

        # Calculate scaling factor
        sm_scale = self._get_sm_scale(q, qpe)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, qpe, kv, kpe, sm_scale)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: tilegym.ops.mla_decoding_split_kv(q, qpe, kv, kpe, sm_scale)
        else:
            pytest.skip(f"Framework {framework} is not available")

        if framework != "pytorch":
            # Verify correctness before benchmarking
            atol = 1e-2
            rtol = 1e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(q, qpe, kv, kpe, sm_scale),
                kwargs={},
                atol=atol,
                rtol=rtol,
                multiple_outputs=False,
            )

        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)
