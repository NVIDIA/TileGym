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


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("offset", [0j, 1 + 0j, 999j])
def test_compare_tensors_complex_components(dtype, chunk_size, offset):
    reference = torch.tensor([1 + 2j, 3 - 4j], dtype=dtype)
    actual = reference.clone()
    actual[0] += offset

    allclose, _ = common.compare_tensors(actual, reference, rtol=0, atol=0, chunk_size=chunk_size)

    assert allclose == (offset == 0j)


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_compare_tensors_preserves_complex128_precision(chunk_size):
    reference = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    actual = reference.clone()
    actual.imag[0] = torch.nextafter(actual.imag[0], torch.tensor(3.0, dtype=torch.float64))

    allclose, _ = common.compare_tensors(actual, reference, rtol=0, atol=0, chunk_size=chunk_size)

    assert not allclose


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_compare_tensors_complex_relative_tolerance(chunk_size):
    reference = torch.tensor([100 + 0j, 200 + 0j], dtype=torch.complex64)
    actual = reference + 0.5j

    allclose, _ = common.compare_tensors(actual, reference, rtol=0.01, atol=0, chunk_size=chunk_size)

    assert allclose


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_compare_tensors_range_labels(dtype, chunk_size):
    tensor = torch.tensor([1, 2], dtype=dtype)

    allclose, messages = common.compare_tensors(tensor, tensor, chunk_size=chunk_size)

    assert allclose
    label = "real range" if dtype.is_complex else "range"
    assert any(message.lstrip().startswith(f"ref {label}:") for message in messages)
    assert any(message.lstrip().startswith(f"test {label}:") for message in messages)
