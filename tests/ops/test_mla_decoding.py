# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
from tests import common
from tilegym.backend import set_backend


class Test_MLADecoding(common.PyTestCase):
    @staticmethod
    def reference(q, qpe, kv, kpe, sm_scale=None, compute_dtype=torch.half):
        """Reference implementation using PyTorch for MLA decoding"""
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
        return o, (m / math.log(2) + torch.log2(l))

    @staticmethod
    def _get_sm_scale(q, qpe):
        """Calculate the default attention scale factor"""
        return 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "num_heads, transpose",
        [(16, True), (32, True), (64, False), (128, False)],
    )
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("BLOCK_D, BLOCK_KPE", [(512, 64)])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        num_heads,
        transpose,
        dtype,
        BLOCK_D,
        BLOCK_KPE,
        backend,
        arch,
    ):
        """Test functional correctness of MLA decoding"""
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")

        if backend == "cutile":
            if not transpose:
                pytest.skip("Skip due to CuTile MLA Decoding only supports transpose=True")

        self.setUp()
        num_heads = 32
        num_batch = 2
        S_kv = 1024
        device = torch.device("cuda")

        # Generate test data
        q = (
            torch.empty(num_batch, num_heads, BLOCK_D, device=device, dtype=torch.float32)
            .normal_(mean=0.3, std=0.2)
            .to(dtype)
        )

        qpe = (
            torch.empty(num_batch, num_heads, BLOCK_KPE, device=device, dtype=torch.float32)
            .normal_(mean=0.3, std=0.1)
            .to(dtype)
            if BLOCK_KPE > 0
            else torch.empty(num_batch, num_heads, 0, device=device, dtype=dtype)
        )

        kv = (
            torch.empty(num_batch, S_kv, BLOCK_D, device=device, dtype=torch.float32)
            .normal_(mean=0.3, std=0.2)
            .to(dtype)
        )

        kpe = (
            torch.empty(num_batch, S_kv, BLOCK_KPE, device=device, dtype=torch.float32)
            .normal_(mean=0.3, std=0.1)
            .to(dtype)
            if BLOCK_KPE > 0
            else torch.empty(num_batch, S_kv, 0, device=device, dtype=dtype)
        )

        # Calculate proper scale factor
        sm_scale = self._get_sm_scale(q, qpe)
        if backend == "cutile":

            def tilegym_fn():
                return tilegym.ops.cutile.mla_decoding.mla_decoding(
                    q,
                    qpe,
                    kv,
                    kpe,
                    sm_scale,
                )

        else:
            pytest.skip(f"Backend {backend} not supported")

        def ref_fn():
            return self.reference(q, qpe, kv, kpe, sm_scale)

        # Set tolerance based on dtype
        rtol = 0.01 if dtype == torch.float16 else 0.02
        atol = 0.01 if dtype == torch.float16 else 0.02

        self.assertCorrectness(
            tilegym_fn,
            ref_fn,
            {},
            rtol=rtol,
            atol=atol,
            multiple_outputs=True,
        )

    @pytest.mark.parametrize(
        "num_heads, transpose",
        [(16, True), (32, True), (64, False), (128, False)],
    )
    @pytest.mark.parametrize("dtype", ["float16", "float8_e5m2"])
    @pytest.mark.parametrize("BLOCK_D, BLOCK_KPE", [(512, 64)])
    @pytest.mark.parametrize("num_batch", [1, 128, 1024])
    @pytest.mark.parametrize("S_kv", [1024, 2048, 8192])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(
        self,
        num_heads,
        transpose,
        dtype,
        BLOCK_D,
        BLOCK_KPE,
        num_batch,
        S_kv,
        framework,
        record_property,
    ):
        """Performance test for MLA decoding with various configurations"""
        # Convert string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e5m2": torch.float8_e5m2,
        }
        dtype = dtype_map[dtype]

        if torch.cuda.get_device_capability()[0] == 8 and dtype == torch.float8_e5m2:
            pytest.skip("Skip due to sm80 not support fp8 type")
        if torch.cuda.get_device_capability() == (12, 0) and S_kv == 8192:
            pytest.skip("Skip OOM on B20X (sm120): MLA decoding with seqlen=8192 exceeds 32 GiB VRAM")

        self.setUp()

        if framework == "cutile":
            if not transpose:
                pytest.skip("Skip due to cutile not support transpose")
            if dtype == torch.float8_e5m2:
                pytest.skip("Skip due to cutile not support float8_e5m2")

        device = torch.device("cuda")
        # Generate test data
        q = torch.ones(num_batch, num_heads, BLOCK_D, device=device, dtype=torch.float32).to(dtype)

        qpe = torch.ones(num_batch, num_heads, BLOCK_KPE, device=device, dtype=torch.float32).to(dtype)

        kv = torch.ones(num_batch, S_kv, BLOCK_D, device=device, dtype=torch.float32).to(dtype)

        kpe = torch.ones(num_batch, S_kv, BLOCK_KPE, device=device, dtype=torch.float32).to(dtype)

        # Calculate proper scale factor
        sm_scale = self._get_sm_scale(q, qpe)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, qpe, kv, kpe, sm_scale)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            if framework == "cutile":
                framework_fn = lambda: tilegym.ops.cutile.mla_decoding.mla_decoding(q, qpe, kv, kpe, sm_scale)
            else:
                pytest.skip(f"Framework {framework} not supported")
        else:
            pytest.skip(f"Framework {framework} is not available")

        # Run benchmarks
        res = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)

        # Record results for reporting
        record_property("benchmark", res)

        # Run after benchmark
        if framework != "pytorch":
            if dtype == torch.float8_e5m2:
                atol = 0.5
                rtol = 0.5
            else:
                atol = 1e-2
                rtol = 1e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(q, qpe, kv, kpe, sm_scale),
                kwargs={},
                atol=atol,
                rtol=rtol,
                multiple_outputs=True,
            )

        # Explicit cleanup to prevent OOM
        del q, qpe, kv, kpe, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
