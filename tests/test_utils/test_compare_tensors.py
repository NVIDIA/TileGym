# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

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
