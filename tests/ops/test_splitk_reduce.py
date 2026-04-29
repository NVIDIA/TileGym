# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
from tests import common


class Test_SplitkReduce(common.PyTestCase):
    @staticmethod
    def reference(attn_splitk_out, lse_splitk_out, S_kv):
        """Reference implementation using PyTorch for splitk reduce"""
        B, num_heads, NUM_KV_SPLITS, head_dim = attn_splitk_out.shape

        # Convert to float32 for computation precision
        attn_splitk = attn_splitk_out.float()
        lse_splitk = lse_splitk_out.float()

        # Find the maximum LSE across splits for numerical stability
        lse_max = torch.max(lse_splitk, dim=-1, keepdim=True)[0]  # [B, num_heads, 1]

        # Compute normalized sum-exp values
        # Convert from log2 to natural log, then back to log2
        sumexp_normalized_splitk = torch.exp2(lse_splitk - lse_max) / math.log(2)
        sumexp_normalized = torch.sum(sumexp_normalized_splitk, dim=-1, keepdim=True)  # [B, num_heads, 1]

        # Weight each split's attention output by its normalized sum-exp
        numerator_normalized = torch.sum(
            attn_splitk * sumexp_normalized_splitk.unsqueeze(-1), dim=2
        )  # [B, num_heads, head_dim]

        # Final output
        attn_out = numerator_normalized / sumexp_normalized  # [B, num_heads, head_dim]

        return attn_out.to(attn_splitk_out.dtype)

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize("batch_size", [1, 2])  # Match MLA workload
    @pytest.mark.parametrize("num_heads", [16, 32])  # Match MLA workload
    @pytest.mark.parametrize("head_dim", [512])  # Match MLA BLOCK_D
    @pytest.mark.parametrize("num_kv_splits", [1, 2, 8, 16])  # Common MLA split counts
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        num_heads,
        head_dim,
        num_kv_splits,
        dtype,
        backend,
        arch,
    ):
        """Test functional correctness of splitk_reduce"""
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping splitk_reduce test")

        device = torch.device("cuda")
        torch.manual_seed(42)  # For reproducibility

        # Set S_kv based on num_kv_splits to match MLA actual workload
        # MLA uses kv_len_per_split of 128 or 512
        if num_kv_splits == 1:
            S_kv = 129  # Small sequence, single split
        elif num_kv_splits == 2:
            S_kv = 256  # Can be from S_kv=129 with kv_len_per_split=128, or 1024 with 512
        elif num_kv_splits == 8:
            S_kv = 1024  # S_kv=1024 with kv_len_per_split=128
        elif num_kv_splits == 16:
            S_kv = 2048  # S_kv=8192 with kv_len_per_split=512, scaled down for testing
        else:
            S_kv = num_kv_splits * 128  # Default: assume kv_len_per_split=128

        # Create realistic intermediate attention results
        # These should represent partial attention outputs from different KV splits
        attn_splitk_out = torch.randn(
            batch_size,
            num_heads,
            num_kv_splits,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # Create LSE values - these should be reasonable log-sum-exp values
        # Initialize with values that would come from actual attention computation
        lse_splitk_out = (
            torch.randn(
                batch_size,
                num_heads,
                num_kv_splits,
                device=device,
                dtype=torch.float32,
            )
            * 2.0
            + 5.0
        )  # Scale to reasonable LSE range

        # Allocate output tensor
        attn_out = torch.empty(batch_size, num_heads, head_dim, device=device, dtype=dtype)

        def splitk_reduce_fn():
            return tilegym.ops.splitk_reduce(attn_splitk_out, lse_splitk_out, attn_out, S_kv)

        def ref_fn():
            return self.reference(attn_splitk_out, lse_splitk_out, S_kv)

        self.assertCorrectness(
            splitk_reduce_fn,
            ref_fn,
            {},
            atol=1e-2,
            rtol=1e-2,
            multiple_outputs=False,
        )

    @pytest.mark.parametrize("batch_size", [1])  # Match MLA workload
    @pytest.mark.parametrize("num_heads", [16, 32])
    @pytest.mark.parametrize("head_dim", [512])  # Match MLA BLOCK_D
    @pytest.mark.parametrize("num_kv_splits", [2, 8, 16, 64])  # MLA common split counts
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(
        self,
        batch_size,
        num_heads,
        head_dim,
        num_kv_splits,
        dtype,
        framework,
        record_property,
    ):
        """Performance test for splitk_reduce"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")

        self.setUp()
        device = torch.device("cuda")
        torch.manual_seed(42)  # For reproducibility

        # Set S_kv based on num_kv_splits to match MLA actual workload
        if num_kv_splits == 2:
            S_kv = 1024  # S_kv=1024 with kv_len_per_split=512
        elif num_kv_splits == 8:
            S_kv = 1024  # S_kv=1024 with kv_len_per_split=128
        elif num_kv_splits == 16:
            S_kv = 8192  # S_kv=8192 with kv_len_per_split=512
        elif num_kv_splits == 64:
            S_kv = 8192  # S_kv=8192 with kv_len_per_split=128
        else:
            S_kv = num_kv_splits * 128  # Default

        # Create test data
        attn_splitk_out = torch.randn(
            batch_size,
            num_heads,
            num_kv_splits,
            head_dim,
            device=device,
            dtype=dtype,
        )
        lse_splitk_out = (
            torch.randn(
                batch_size,
                num_heads,
                num_kv_splits,
                device=device,
                dtype=torch.float32,
            )
            * 2.0
            + 5.0
        )

        attn_out = torch.empty(batch_size, num_heads, head_dim, device=device, dtype=dtype)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(attn_splitk_out, lse_splitk_out, S_kv)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: tilegym.ops.splitk_reduce(attn_splitk_out, lse_splitk_out, attn_out, S_kv)
        else:
            pytest.skip(f"Framework {framework} is not available")

        if framework != "pytorch":
            # Verify correctness before benchmarking
            atol = 1e-2
            rtol = 1e-2
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(attn_splitk_out, lse_splitk_out, S_kv),
                kwargs={},
                atol=atol,
                rtol=rtol,
                multiple_outputs=False,
            )

        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del attn_splitk_out, lse_splitk_out, attn_out, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
