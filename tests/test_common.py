# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch

from . import common


def test_compare_tensors_preserves_float64_precision():
    # nextafter gives `actual` a 1-ULP (float64) offset from `reference` — a gap that
    # would vanish under a float32 cast, so this fails unless compare_tensors keeps
    # float64 precision.
    reference = torch.tensor([1.0], dtype=torch.float64)
    actual = torch.nextafter(reference, torch.tensor([2.0], dtype=torch.float64))

    allclose, _ = common.compare_tensors(actual, reference, rtol=0.0, atol=0.0)

    assert not allclose
