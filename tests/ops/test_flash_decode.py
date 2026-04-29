# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym

from .. import common


class Test_FlashDecode(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, sm_scale):
        torch.backends.cuda.mem_efficient_sdp_enabled()
        ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, enable_gqa=True)

        return ref_output

    _backends = ["cutile"]
    _perf_frameworks = _backends + ["pytorch"]

    @pytest.mark.parametrize("seq_len", [9, 119, 256, 8192])
    @pytest.mark.parametrize("group_size", [1, 4, 8])
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, seq_len, group_size, dtype, backend, arch):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping FMHA test")

        # Test parameters
        batch_size = 2
        num_heads = 32
        head_dim = 64

        # Create random input tensors
        torch.manual_seed(42)  # For reproducibility
        q = torch.randn(batch_size, num_heads, 1, head_dim, device="cuda").to(dtype)
        k = torch.randn(
            batch_size,
            num_heads // group_size,
            seq_len,
            head_dim,
            device="cuda",
        ).to(dtype)
        v = torch.randn(
            batch_size,
            num_heads // group_size,
            seq_len,
            head_dim,
            device="cuda",
        ).to(dtype)

        # Compute softmax scale
        sm_scale = 1.0 / math.sqrt(head_dim)

        self.assertCorrectness(
            tilegym.ops.fmha_decode,
            self.reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "sm_scale": sm_scale,
            },
            atol=1e-2,
            rtol=1e-2,
            check_stride=False,
        )

    @pytest.mark.parametrize(
        "batch_size, num_heads, seq_len, head_dim, group_size, dtype",
        [
            (batch_size, num_heads, seq_len, head_dim, group_size, dtype)
            for batch_size in [1]
            for num_heads in [32]
            for head_dim in [128]
            for dtype in [torch.float16]
            for seq_len in [
                2**9,  # 512
                2**10,  # 1024
                2**11,  # 2048
                2**12,  # 4096
                2**13,  # 8192
                2**14,  # 16384
            ]
            + [31079]  # benchmark size in llama model
            for group_size in [
                1,
                4,
                8,
            ]  # when group_size = 1, it' the naive case, otherwise it's the grouped case
        ],
        ids=lambda x: (str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x)),
    )
    @pytest.mark.parametrize("framework", _perf_frameworks)
    def test_perf(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        group_size,
        dtype,
        framework,
        record_property,
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability()[0] == 12:
            pytest.xfail("Shared memory exhaustion on sm120: FlashDecode requires 133,152 B > hardware limit 102,400 B")

        self.setUp()
        device = torch.device("cuda")

        # Create test data with specified dtype
        torch.manual_seed(42)  # For reproducibility
        q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
        k = torch.randn(
            batch_size,
            num_heads // group_size,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v = torch.randn(
            batch_size,
            num_heads // group_size,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)

        if framework == "pytorch":
            framework_fn = lambda: self.reference(q, k, v, sm_scale)
        elif tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
            framework_fn = lambda: tilegym.ops.fmha_decode(q, k, v, sm_scale)  # , kv_len_per_split=2048 * 2
        else:
            pytest.skip(f"Framework {framework} is not available")

        if framework != "pytorch":
            # Verify correctness before benchmarking
            atol = 1e-3
            rtol = 1e-3
            self.assertCorrectness(
                framework_fn,
                lambda: self.reference(q, k, v, sm_scale),
                kwargs={},
                atol=atol,
                rtol=rtol,
                check_stride=False,
            )
        result = common.benchmark_framework(framework, framework_fn, use_cudagraph=True)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del q, k, v, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
