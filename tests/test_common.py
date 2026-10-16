# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
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


def test_benchmark_framework_cools_down_before_warmup(monkeypatch):
    calls = []
    framework_fn = object()

    monkeypatch.setattr(common.torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(common.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))

    def benchmark(fn, **kwargs):
        calls.append(("benchmark", fn, kwargs))
        return "result"

    monkeypatch.setattr(common, "benchmark", benchmark)

    result = common.benchmark_framework("sota", framework_fn, cooldown_s=1.0, warmup=100)

    assert result == {"sota": "result"}
    assert calls == ["synchronize", ("sleep", 1.0), ("benchmark", framework_fn, {"warmup": 100})]


def test_benchmark_framework_skips_zero_cooldown(monkeypatch):
    monkeypatch.setattr(common.torch.cuda, "synchronize", lambda: pytest.fail("unexpected synchronization"))
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: pytest.fail("unexpected sleep"))
    monkeypatch.setattr(common, "benchmark", lambda _fn, **_kwargs: "result")

    assert common.benchmark_framework("cutile", object()) == {"cutile": "result"}


def test_benchmark_framework_rejects_negative_cooldown():
    with pytest.raises(ValueError, match="cooldown_s must be non-negative"):
        common.benchmark_framework("sota", object(), cooldown_s=-1.0)
