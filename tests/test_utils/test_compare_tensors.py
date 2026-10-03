# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

from tests import common


def test_chunked_compare_matches_full_diagnostics():
    reference = torch.tensor(
        [
            [[1.0, -2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0]],
        ],
        dtype=torch.bfloat16,
    )
    test = reference.clone()
    test[1, 0, 1] = 6.5
    test[4, 1, 0] = 21.0

    full_result = common.compare_tensors(test, reference, rtol=1e-2, atol=1e-2, msg_prefix=None)
    chunked_result = common.compare_tensors(
        test,
        reference,
        rtol=1e-2,
        atol=1e-2,
        msg_prefix=None,
        chunk_size=2,
    )

    assert chunked_result == full_result


def test_chunked_compare_rejects_nonpositive_chunk_size():
    tensor = torch.ones(2)

    for chunk_size in (0, -1):
        try:
            common.compare_tensors(tensor, tensor, chunk_size=chunk_size)
        except ValueError as error:
            assert str(error) == f"chunk_size must be positive, got {chunk_size}"
        else:
            raise AssertionError("compare_tensors accepted a nonpositive chunk size")


@pytest.mark.parametrize("shape", [(1, 19), (2, 19), (1, 1, 19), (2, 3, 19), (7, 3)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_comparison_splits_wide_rows_with_original_diagnostics(monkeypatch, shape, strided, chunk_size):
    storage = torch.arange(math.prod(shape) * 2, dtype=torch.float64).reshape(*shape, 2) - 5
    reference = storage[..., 0] if strided else storage[..., 0].contiguous()
    test_storage = storage.clone()
    test = test_storage[..., 0] if strided else test_storage[..., 0].contiguous()
    test[(0,) * len(shape)] += 0.25
    test[(-1,) * len(shape)] += 1e-9
    expected = common.compare_tensors(test, reference, rtol=0, atol=1e-10, msg_prefix=None)
    allclose = torch.allclose
    chunk_elements = []

    def bounded_allclose(actual, expected, *args, **kwargs):
        chunk_elements.append(actual.numel())
        assert actual.numel() <= 8
        return allclose(actual, expected, *args, **kwargs)

    monkeypatch.setattr(common, "_COMPARISON_CHUNK_ELEMENTS", 8)
    monkeypatch.setattr(torch, "allclose", bounded_allclose)
    actual = common.compare_tensors(test, reference, rtol=0, atol=1e-10, msg_prefix=None, chunk_size=chunk_size)
    assert not actual[0]
    assert sum(chunk_elements) == test.numel()
    assert actual[1][:-2] == expected[1][:-2]
    assert actual[1][-1] == expected[1][-1]
    assert actual[1][-2] == f"shape: {test.shape} stride: {test.stride()} dtype: {test.dtype}"
