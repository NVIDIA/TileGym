# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import random
import sys

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites import flashinfer


def create_ragged_m_segments(num_groups, m, dtype, align_to=None):
    """Create non-even M segments for ragged BMM

    Args:
        num_groups: Number of batches/groups
        m: Average segment size
        dtype: Data type
        align_to: If specified, align segment sizes to this value
    """
    total_m = num_groups * m
    segment_sizes = []
    itemsize = dtype.itemsize
    num_items = 16 // itemsize

    # Use align_to if specified, otherwise use default alignment
    alignment = align_to if align_to is not None else num_items

    # Generate random segment sizes
    for i in range(num_groups - 1):
        size = int(m * random.uniform(0.5, 1.5))
        size = (size // alignment) * alignment
        if size < alignment:
            size = alignment
        segment_sizes.append(size)

    remaining = total_m - sum(segment_sizes)
    remaining = (remaining // alignment) * alignment
    if remaining < alignment:
        remaining = alignment
    segment_sizes.append(remaining)

    actual_total_m = sum(segment_sizes)

    segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        segment_offsets[i + 1] = segment_offsets[i] + segment_sizes[i]

    max_m = max(segment_sizes)
    return max_m, segment_offsets, actual_total_m


class Test_FlashInfer_RaggedBMM(common.PyTestCase):
    @staticmethod
    def reference(a, b, segment_offsets, trans_a=False, trans_b=True, out_dtype=None):
        """
        PyTorch reference for ragged BMM with non-even M segments.
        Matrix a is flattened with segment_offsets defining the boundaries.
        """
        if trans_a:
            a = torch.transpose(a, 0, 1)
        if trans_b:
            b = torch.transpose(b, 1, 2)

        total_m, K = a.shape
        Q, K_b, N = b.shape

        if out_dtype is None:
            out_dtype = a.dtype

        c = torch.zeros((total_m, N), device=a.device, dtype=out_dtype)

        for q in range(Q):
            start_offset = segment_offsets[q].item()
            end_offset = segment_offsets[q + 1].item()
            segment_size = end_offset - start_offset
            assert segment_size > 0
            a_segment = a[start_offset:end_offset, :]
            b_segment = b[q, :, :]
            c_segment = torch.mm(a_segment.to(out_dtype), b_segment.to(out_dtype))
            c[start_offset:end_offset, :] = c_segment

        return c

    @staticmethod
    def prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype, framework="cutile"):
        device = torch.device("cuda")

        # For CuTile, we need segments aligned to BLOCK_M (128)
        # This ensures segment offsets are multiples of the tile size
        align_to = 128 if framework == "cutile" else None

        max_m, segment_offsets, actual_total_m = create_ragged_m_segments(
            num_groups=num_groups,
            m=m,
            dtype=dtype,
            align_to=align_to,
        )

        total_m = segment_offsets[-1].item()

        if trans_a:
            a_shape = (k, total_m)
        else:
            a_shape = (total_m, k)

        if trans_b:
            b_shape = (num_groups, n, k)
        else:
            b_shape = (num_groups, k, n)

        a = torch.rand(a_shape, device=device, dtype=torch.float16, requires_grad=False).to(dtype)
        b = torch.rand(b_shape, device=device, dtype=torch.float16, requires_grad=False).to(dtype)

        return a, b, max_m, segment_offsets

    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize("dtype", [(torch.bfloat16)])
    @pytest.mark.parametrize("num_groups, m, n, k", [(4, 256, 256, 256), (2, 128, 128, 128), (4, 512, 512, 512)])
    def test_op_shapes(self, framework, trans_a, trans_b, dtype, num_groups, m, n, k):
        # cutile kernel only supports (trans_a=False, trans_b=True)
        if framework == "cutile" and (trans_a or not trans_b):
            pytest.skip("ragged_bmm only supports trans_a=False, trans_b=True")
        _impl_fw = ["cutile"]
        if framework not in _impl_fw:
            pytest.skip(f"Framework {framework} not supported")
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(0)
        random.seed(0)
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype, framework)

        framework_fn = lambda: flashinfer.ops.ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
        )
        self.assertCorrectness(
            framework_fn,
            lambda: self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    @pytest.mark.parametrize("dtype", [(torch.bfloat16)])
    @pytest.mark.parametrize("m, n, k", [(256, 256, 256)])
    @pytest.mark.parametrize("num_groups", [1, 4, 8])
    def test_op_num_groups(self, framework, dtype, m, n, k, num_groups):
        _impl_fw = ["cutile"]
        if framework not in _impl_fw:
            pytest.skip(f"Framework {framework} not supported")
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(0)
        random.seed(0)
        trans_a = False
        trans_b = True
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype, framework)

        framework_fn = lambda: flashinfer.ops.ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
        )
        self.assertCorrectness(
            framework_fn,
            lambda: self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    @pytest.mark.parametrize("num_groups, m, n, k", [(4, 256, 256, 256)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
    def test_op_dtypes(self, framework, num_groups, m, n, k, dtype, arch):
        _impl_fw = ["cutile"]
        if framework not in _impl_fw:
            pytest.skip(f"Framework {framework} not supported")
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        if arch == "sm80" and "float8" in dtype.__repr__():
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        torch.manual_seed(0)
        random.seed(0)
        trans_a = False
        trans_b = True
        out_dtype = torch.bfloat16
        a, b, max_m, segment_offsets = self.prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype, framework)

        framework_fn = lambda: flashinfer.ops.ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
        )
        self.assertCorrectness(
            framework_fn,
            lambda: self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    @pytest.mark.parametrize("dtype", [(torch.bfloat16)])
    @pytest.mark.parametrize("num_groups, m, n, k", [(4, 256, 256, 256)])
    @pytest.mark.parametrize("trans_a", [False, True])
    @pytest.mark.parametrize("trans_b", [False, True])
    def test_op_transpose(self, framework, dtype, num_groups, m, n, k, trans_a, trans_b):
        # cutile kernel only supports (trans_a=False, trans_b=True)
        if framework == "cutile" and (trans_a or not trans_b):
            pytest.skip("ragged_bmm only supports trans_a=False, trans_b=True")
        _impl_fw = ["cutile"]
        if framework not in _impl_fw:
            pytest.skip(f"Framework {framework} not supported")
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")

        torch.manual_seed(0)
        random.seed(0)
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype, framework)

        framework_fn = lambda: flashinfer.ops.ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
        )
        self.assertCorrectness(
            framework_fn,
            lambda: self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype),
            kwargs={},
            atol=1e-2,
            rtol=1e-2,
        )
