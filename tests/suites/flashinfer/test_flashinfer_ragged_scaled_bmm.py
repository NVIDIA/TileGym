# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import random

import pytest
import torch
from triton.tools.mxfp import MXFP4Tensor
from triton.tools.mxfp import MXScaleTensor

import tilegym
from tests import common
from tilegym.suites import flashinfer

_backends = ["cutile"]
_perf_backends = _backends


def create_ragged_m_segments(num_groups, m, ELEM_PER_BYTE_A):
    """Create non-even M segments for ragged BMM"""
    # Create random segment sizes that sum to approximately total_m
    total_m = num_groups * m
    segment_sizes = []
    num_items = 16 * ELEM_PER_BYTE_A
    alignment_sf = 128

    # Generate random segment sizes
    for i in range(num_groups - 1):
        # Random size between 0.5x and 1.5x expected size
        size = int(m * random.uniform(0.5, 1.5))
        size = (size // num_items) * num_items
        size = (size // alignment_sf) * alignment_sf
        segment_sizes.append(size)

    # Last segment gets the remaining size
    remaining = total_m - sum(segment_sizes)
    assert remaining > 0 and remaining % num_items == 0 and remaining % alignment_sf == 0
    segment_sizes.append(remaining)

    # Create segment offsets
    segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        segment_offsets[i + 1] = segment_offsets[i] + segment_sizes[i]

    max_m = max(segment_sizes)

    return max_m, segment_offsets


class Test_FlashInfer_RaggedScaledBMM(common.PyTestCase):
    @staticmethod
    def reference(a, b, segment_offsets, trans_a=False, trans_b=True):
        """
        PyTorch reference for ragged BMM with non-even M segments.
        Matrix a is flattened with segment_offsets defining the boundaries.
        """

        # Get dimensions
        total_m, K = a.shape
        Q, K_b, N = b.shape

        # Initialize output tensor
        c = torch.zeros((total_m, N), device=a.device, dtype=a.dtype)

        # Process each segment
        for q in range(Q):
            start_offset = segment_offsets[q].item()
            end_offset = segment_offsets[q + 1].item()
            segment_size = end_offset - start_offset
            assert segment_size > 0

            # Extract segment from flattened matrix a
            a_segment = a[start_offset:end_offset, :]  # Shape: [segment_size, K]

            b_segment = b[q, :, :]  # Shape: [K, N]

            # Compute matrix multiplication for this segment
            c_segment = torch.mm(a_segment, b_segment)

            # Store the result in the output tensor
            c[start_offset:end_offset, :] = c_segment

        return c

    @staticmethod
    def initialize_block_scaled(
        num_groups,
        M,
        N,
        K,
        block_scale_type="nvfp4",
        compute_reference=False,
        trans_a=False,
        trans_b=True,
    ):
        VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
        assert block_scale_type in [
            "nvfp4",
            "mxfp4",
            "mxfp8",
            "mixed",
        ], f"Invalid block scale type: {block_scale_type}"

        Q = num_groups
        assert N % 8 == 0, "N must be divisible by 8"
        assert K % 128 == 0, "K must be divisible by 128"
        assert trans_a == False and trans_b == True, "Only NT layout is supported"
        ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
        # Create ragged M segments
        max_m, segment_offsets = create_ragged_m_segments(
            num_groups=num_groups,
            m=M,
            ELEM_PER_BYTE_A=ELEM_PER_BYTE_A,
        )

        total_m = segment_offsets[-1].item()
        assert total_m == num_groups * M

        device = "cuda"
        a_ref = MXFP4Tensor(size=(total_m, K), device=device).random()
        # B is [Q, N, K] (K-major, the NT layout this op takes). fp4 is packed along K
        # below, and b_ref is transposed to the logical (Q, K, N) operand.
        b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
        if block_scale_type in ["mxfp8", "mixed"]:
            a_ref = a_ref.to(torch.float32)
            a = a_ref.to(torch.float8_e4m3fn)
        else:
            # Pack two fp4 elements per byte along K
            a = a_ref.to_packed_tensor(dim=1)

        if block_scale_type == "mxfp8":
            b_ref = b_ref.to(torch.float32)
            b = b_ref.to(torch.float8_e4m3fn)
        else:
            b = b_ref.to_packed_tensor(dim=2)

        b_ref = b_ref.to(torch.float32)
        b_ref = torch.transpose(b_ref, 1, 2)

        # Focus on NT layout, so k is the leading dimension of both A and B.
        alignment_sf = 128
        total_m_padded = (total_m + (alignment_sf - 1) * num_groups) // alignment_sf * alignment_sf
        total_m_padded = total_m
        assert total_m_padded == total_m
        a_scale_shape = [total_m_padded // 128, K // VEC_SIZE // 4, 32, 16]
        n_padded = (N + alignment_sf - 1) // alignment_sf * alignment_sf
        n_padded = N
        assert n_padded == N
        b_scale_shape = [Q, n_padded // 128, K // VEC_SIZE // 4, 32, 16]
        epsilon = 1e-8
        a_scale = torch.rand(a_scale_shape, device=device) + epsilon
        b_scale = torch.rand(b_scale_shape, device=device) + epsilon
        if block_scale_type == "nvfp4":
            a_scale = a_scale.to(torch.float8_e4m3fn)
            b_scale = b_scale.to(torch.float8_e4m3fn)
            a_scale_ref = a_scale
            b_scale_ref = b_scale
        elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
            a_scale_ref = MXScaleTensor(a_scale)
            b_scale_ref = MXScaleTensor(b_scale)
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data

        a_scale = a_scale.reshape(a_scale_shape[0], a_scale.shape[1], 2, 256)
        b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)

        c = None
        if compute_reference:
            packed_a_scale_ref = (
                a_scale_ref.to(torch.float32)
                .reshape(a_scale_shape[0], a_scale.shape[1], 32, 4, 4)
                .permute(0, 3, 2, 1, 4)
                .reshape(a_scale_shape[0] * 128, a_scale.shape[1] * 4)
                .contiguous()
            )
            unpacked_a_scale_ref = packed_a_scale_ref.repeat_interleave(VEC_SIZE, dim=1).contiguous()[
                :total_m_padded, :K
            ]

            packed_b_scale_ref = (
                b_scale_ref.to(torch.float32)
                .reshape(Q, b_scale_shape[1], b_scale.shape[2], 32, 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .reshape(Q, b_scale_shape[1] * 128, b_scale.shape[2] * 4)
                .contiguous()
            )
            unpacked_b_scale_ref = (
                packed_b_scale_ref.repeat_interleave(VEC_SIZE, dim=2).permute(0, 2, 1).contiguous()[:Q, :K, :n_padded]
            )
            a_ref_float = a_ref.to(torch.float32)

            c = Test_FlashInfer_RaggedScaledBMM.reference(
                a_ref_float * unpacked_a_scale_ref,
                b_ref * unpacked_b_scale_ref,
                segment_offsets,
                trans_a,
                trans_b,
            )

        return a, b, a_scale, b_scale, segment_offsets, max_m, c

    @pytest.mark.parametrize("num_groups", [3])
    @pytest.mark.parametrize("m", [512])
    @pytest.mark.parametrize("n", [2048])
    @pytest.mark.parametrize("k", [1024, 4096])
    @pytest.mark.parametrize("block_scale_type", ["mxfp8"])
    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [True])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, num_groups, m, n, k, block_scale_type, trans_a, trans_b, backend):
        if torch.cuda.get_device_capability() < (10, 0):
            pytest.skip("mxfp8 ragged_scaled_bmm targets SM100+ GPUs")
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        self.setUp()
        torch.manual_seed(0)
        random.seed(0)
        (
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            ref_c,
        ) = self.initialize_block_scaled(
            num_groups,
            m,
            n,
            k,
            block_scale_type,
            True,
            trans_a,
            trans_b,
        )

        c = flashinfer.ops.ragged_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            block_scale_type,
            trans_a,
            trans_b,
        )

        torch.testing.assert_close(ref_c, c.to(torch.float32), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("num_groups", [3])
    @pytest.mark.parametrize("m", [512, 1024, 4096])
    @pytest.mark.parametrize("n", [2048, 4096])
    @pytest.mark.parametrize("k", [1024, 4096])
    @pytest.mark.parametrize("block_scale_type", ["mxfp8"])
    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [True])
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(
        self,
        num_groups,
        m,
        n,
        k,
        block_scale_type,
        trans_a,
        trans_b,
        backend,
        record_property,
    ):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        torch.manual_seed(0)
        random.seed(0)
        (
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            ref_c,
        ) = self.initialize_block_scaled(
            num_groups,
            m,
            n,
            k,
            block_scale_type,
            True,
            trans_a,
            trans_b,
        )

        backend_fn = lambda: flashinfer.ops.ragged_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            block_scale_type,
            trans_a,
            trans_b,
        )

        res = common.benchmark_framework(backend, backend_fn)
        record_property("benchmark", res)

        c = flashinfer.ops.ragged_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            block_scale_type,
            trans_a,
            trans_b,
        )

        torch.testing.assert_close(ref_c, c.to(torch.float32), atol=1e-2, rtol=1e-2)
