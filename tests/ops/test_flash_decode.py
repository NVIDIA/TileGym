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
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, enable_gqa=True
        )

        return ref_output

    _backends = ["cutile"]

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
                'q': q,
                'k': k,
                'v': v,
                'sm_scale': sm_scale,
            },
            atol=1e-2,
            rtol=1e-2,
            check_stride=False,
        )
