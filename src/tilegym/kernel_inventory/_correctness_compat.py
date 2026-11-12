# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Output comparison for kernel-inventory runtime checks.

Default allclose plus matched-ratio checks with non-finite and all-zero guards,
or a Definition-provided correctness function that receives cloned outputs and
must return one ``Correctness`` record per compared output.
"""

from __future__ import annotations

import copy
import math
import traceback
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

_DEFAULT_COMPARE_CHUNK_ELEMENTS = 16 * 1024 * 1024


class Correctness(BaseModel):
    """Per-output comparison verdict and checker metadata."""

    output_name: str = Field(default="output", min_length=1)
    passed: bool = Field(default=True)
    has_nan: bool = Field(default=False)
    has_inf: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _move_legacy_error_fields_into_metadata(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        metadata = dict(data.get("metadata") or {})
        for key in ("max_relative_error", "max_absolute_error"):
            if key in data:
                metadata[key] = data.pop(key)
        data["metadata"] = metadata
        return data

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        message = value.get("message")
        if message is not None and not isinstance(message, str):
            raise ValueError("message must be a string")
        for key in ("max_relative_error", "max_absolute_error"):
            if key not in value:
                continue
            number = value[key]
            if not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"{key} must be a finite number")
            if number < 0:
                raise ValueError(f"{key} must be non-negative")
            value[key] = float(number)
        return value


def _iter_tensor_chunks(actual: Any, expected: Any, max_elements: int = _DEFAULT_COMPARE_CHUNK_ELEMENTS) -> Any:
    if actual.is_contiguous() and expected.is_contiguous():
        actual_flat = actual.view(-1)
        expected_flat = expected.view(-1)
        for start in range(0, actual_flat.numel(), max_elements):
            stop = min(actual_flat.numel(), start + max_elements)
            yield actual_flat[start:stop], expected_flat[start:stop]
        return
    if actual.dim() == 0:
        yield actual, expected
        return
    rows = actual.shape[0]
    if rows == 0:
        return
    rows_per_chunk = max(1, max_elements // max(1, actual[0].numel()))
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        yield actual[start:stop], expected[start:stop]


def _check_tensor_sanity(actual: Any, expected: Any, output_name: str, allow_negative_inf: bool) -> Correctness | None:
    import torch

    has_nonfinite = False
    has_nan = False
    expected_norm_sq = 0.0
    actual_norm_sq = 0.0
    for actual_chunk, expected_chunk in _iter_tensor_chunks(actual, expected):
        actual_chunk = actual_chunk.to(torch.float32)
        expected_chunk = expected_chunk.to(torch.float32)
        expected_nonfinite = ~torch.isfinite(expected_chunk)
        actual_nonfinite = ~torch.isfinite(actual_chunk)
        if allow_negative_inf:
            both_negative_inf = (expected_chunk == float("-inf")) & (actual_chunk == float("-inf"))
            expected_nonfinite = expected_nonfinite & ~both_negative_inf
            actual_nonfinite = actual_nonfinite & ~both_negative_inf
        if expected_nonfinite.any().item() or actual_nonfinite.any().item():
            has_nonfinite = True
        if torch.isnan(actual_chunk).any().item() or torch.isnan(expected_chunk).any().item():
            has_nan = True
        if not has_nonfinite:
            expected_norm = float(torch.linalg.vector_norm(expected_chunk).item())
            actual_norm = float(torch.linalg.vector_norm(actual_chunk).item())
            expected_norm_sq += expected_norm * expected_norm
            actual_norm_sq += actual_norm * actual_norm

    if has_nonfinite:
        return Correctness(output_name=output_name, passed=False, has_nan=has_nan, has_inf=not has_nan)
    if expected_norm_sq > 0 and actual_norm_sq == 0:
        error = expected_norm_sq**0.5
        return Correctness(
            output_name=output_name,
            passed=False,
            metadata={"max_absolute_error": error, "max_relative_error": error},
        )
    return None


def compute_allclose_error_stats(
    actual: Any, expected: Any, tolerance: Any, output_name: str = "output"
) -> Correctness:
    """Return the allclose plus matched-ratio verdict for one output."""
    import torch

    allow_negative_inf = tolerance.allow_negative_inf
    sanity = _check_tensor_sanity(actual, expected, output_name, allow_negative_inf)
    if sanity is not None:
        return sanity

    total_elements = 0
    exceeds_count = 0.0
    max_abs = 0.0
    max_rel = 0.0
    for x, y in _iter_tensor_chunks(actual, expected):
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        if allow_negative_inf:
            finite_mask = ~((x == float("-inf")) & (y == float("-inf")))
            if not finite_mask.any().item():
                continue
            x = x[finite_mask]
            y = y[finite_mask]
        abs_error = torch.abs(x - y)
        total_elements += abs_error.numel()
        if abs_error.numel() == 0:
            continue
        max_abs = max(max_abs, float(abs_error.max().item()))
        bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
        exceeds_count += float(((abs_error > bound) | ~torch.isfinite(abs_error)).sum().item())
        rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
        max_rel = max(max_rel, float(rel_error.max().item()))

    if total_elements == 0:
        return Correctness(output_name=output_name, passed=True)
    matched_ratio = max(0.0, min(1.0, 1.0 - exceeds_count / float(total_elements)))
    exceeds = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds = True
    return Correctness(
        output_name=output_name,
        passed=not exceeds,
        metadata={"max_absolute_error": max_abs, "max_relative_error": max_rel},
    )


def validate_custom_correctness_result(result: Any, output_names: list[str], entrypoint_name: str) -> list[Correctness]:
    """Validate a custom correctness return value into ordered ``Correctness`` records."""
    if not isinstance(result, dict):
        raise ValueError(
            f"{entrypoint_name}() must return dict[str, dict] keyed by output name; got {type(result).__name__}"
        )
    expected = set(output_names)
    actual = set(result)
    if actual != expected:
        details = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise ValueError(f"{entrypoint_name}() must return correctness for exactly {output_names}{suffix}")

    correctness = []
    for output_name in output_names:
        data = result[output_name]
        if not isinstance(data, dict):
            raise ValueError(
                f"{entrypoint_name}()[{output_name!r}] must be a dict matching Correctness; got {type(data).__name__}"
            )
        data = dict(data)
        declared_name = data.get("output_name")
        if declared_name is not None and declared_name != output_name:
            raise ValueError(f"{entrypoint_name}()[{output_name!r}] has mismatched output_name {declared_name!r}")
        data["output_name"] = output_name
        try:
            correctness.append(Correctness.model_validate(data))
        except Exception as exc:
            raise ValueError(f"{entrypoint_name}()[{output_name!r}] returned invalid Correctness data: {exc}") from exc
    return correctness


def compare_outputs(
    ref_outputs: list[Any],
    user_outputs: list[Any],
    tolerance: Any,
    output_names: list[str],
    correctness_fn: Any | None = None,
    entrypoint_name: str | None = None,
    *,
    custom_correctness_kwargs: Mapping[str, Any] | None = None,
) -> list[Correctness]:
    """Compare outputs with ``correctness_fn`` or the default allclose checker."""
    if correctness_fn is None:
        if custom_correctness_kwargs:
            raise ValueError("custom_correctness_kwargs requires a custom_correctness_entrypoint")
        return [
            compute_allclose_error_stats(user_output, ref_output, tolerance, output_name=output_name)
            for output_name, ref_output, user_output in zip(output_names, ref_outputs, user_outputs, strict=True)
        ]

    if entrypoint_name is None:
        entrypoint_name = getattr(correctness_fn, "__name__", "custom_correctness")
    ref_for_compare = {name: output.clone() for name, output in zip(output_names, ref_outputs, strict=True)}
    user_for_compare = {name: output.clone() for name, output in zip(output_names, user_outputs, strict=True)}
    try:
        raw_correctness = correctness_fn(
            ref_for_compare,
            user_for_compare,
            **copy.deepcopy(dict(custom_correctness_kwargs or {})),
        )
    except Exception as exc:
        raise ValueError(f"{entrypoint_name}() raised: {exc}\n{traceback.format_exc()}") from exc
    return validate_custom_correctness_result(raw_correctness, output_names, entrypoint_name)


def poison_output_(tensor: Any) -> Any:
    """Fill a destination-passing output with the harness poison in place.

    Floating dtypes get every bit set (NaN), signed integers their minimum,
    unsigned integers their maximum, and bool ``True``.
    """
    import torch

    if tensor.numel() == 0:
        return tensor
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    flat = tensor.as_strided((span,), (1,))
    if tensor.dtype == torch.bool:
        flat.fill_(True)
    elif not tensor.dtype.is_floating_point and not tensor.dtype.is_complex:
        info = torch.iinfo(tensor.dtype)
        flat.fill_(info.min if info.min < 0 else info.max)
    else:
        flat.view(torch.uint8).fill_(0xFF)
    return tensor


__all__ = [
    "Correctness",
    "compare_outputs",
    "compute_allclose_error_stats",
    "poison_output_",
    "validate_custom_correctness_result",
]
