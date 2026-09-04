# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Public compatibility models for the Kernel Factory Workload contract.

Internal builds validate against the pinned canonical schema directly.
Public TileGym releases cannot depend on that private package, so this module
implements the documented Workload subset used by the checked-in inventory.
Project-specific file and Definition compatibility rules intentionally live
outside these schema models.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import Field
from pydantic import StrictBool
from pydantic import StrictFloat
from pydantic import StrictInt
from pydantic import field_validator
from pydantic import model_serializer
from pydantic import model_validator


class EvalMode(str, Enum):
    """Evaluation phases requested for one Workload."""

    FULL = "full"
    CORRECTNESS_ONLY = "correctness_only"
    BENCHMARK_ONLY = "benchmark_only"


class SamplingStrategy(str, Enum):
    """Sampling strategy for correctness-only dynamic axes."""

    RANDOM = "random"
    LINEAR = "linear"


class DynamicAxis(BaseModel):
    """Correctness-only dynamic shape sampling policy for one axis."""

    min: int | str
    max: int | str
    multiple_of: int | str = 1
    sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    intermediate_samples: int = Field(default=0, ge=0)

    @field_validator("min", "max")
    @classmethod
    def _validate_bound(cls, value: int | str) -> int | str:
        if isinstance(value, bool) or isinstance(value, int) and value < 0:
            raise ValueError("dynamic axis bounds must be non-negative integers or const-axis names")
        if isinstance(value, str) and not value:
            raise ValueError("dynamic axis const-axis names must be non-empty")
        return value

    @field_validator("multiple_of")
    @classmethod
    def _validate_multiple_of(cls, value: int | str) -> int | str:
        if isinstance(value, bool) or isinstance(value, int) and value <= 0:
            raise ValueError("dynamic axis multiple_of must be positive")
        if isinstance(value, str) and not value:
            raise ValueError("dynamic axis multiple_of const-axis name must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_numeric_bounds(self) -> "DynamicAxis":
        if isinstance(self.min, int) and isinstance(self.max, int) and self.min > self.max:
            raise ValueError("dynamic axis min must not exceed max")
        return self


class RandomInput(BaseModel):
    """Random tensor input descriptor."""

    type: Literal["random"] = "random"


ScalarValue: TypeAlias = StrictInt | StrictFloat | StrictBool


class ScalarInput(BaseModel):
    """Python numeric scalar input descriptor."""

    type: Literal["scalar"] = "scalar"
    value: ScalarValue


class SafetensorsShard(BaseModel):
    """One rank-local tensor locator."""

    path: str = Field(min_length=1)
    tensor_key: str = Field(min_length=1)


class SafetensorsInput(BaseModel):
    """Replicated or rank-sharded safetensors input descriptor."""

    type: Literal["safetensors"] = "safetensors"
    path: str | None = None
    tensor_key: str | None = None
    shards: list[SafetensorsShard] | None = None

    @model_validator(mode="after")
    def _validate_locator(self) -> "SafetensorsInput":
        has_path = self.path is not None
        has_key = self.tensor_key is not None
        has_shards = self.shards is not None
        if has_path != has_key:
            raise ValueError("safetensors path and tensor_key must be specified together")
        if has_shards and (has_path or has_key):
            raise ValueError("safetensors replicated locator and shards are mutually exclusive")
        if not has_shards and not has_path:
            raise ValueError("safetensors input requires path/tensor_key or shards")
        if has_path and (not self.path or not self.tensor_key):
            raise ValueError("safetensors path and tensor_key must be non-empty")
        if has_shards and not self.shards:
            raise ValueError("safetensors shards must be non-empty")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_unused_locators(self, handler: Any) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


class NullInput(BaseModel):
    """Absent optional input descriptor."""

    type: Literal["null"] = "null"


class StringInput(BaseModel):
    """Python string input descriptor."""

    type: Literal["string"] = "string"
    value: str


class CustomInput(BaseModel):
    """Definition-provided custom input descriptor."""

    type: Literal["custom"] = "custom"


InputSpec: TypeAlias = Annotated[
    RandomInput | ScalarInput | SafetensorsInput | NullInput | StringInput | CustomInput,
    Field(discriminator="type"),
]


class ToleranceSpec(BaseModel):
    """Numerical correctness bounds for one Workload."""

    max_atol: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    max_rtol: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    required_matched_ratio: float = Field(default=0.99, ge=0.0, le=1.0, allow_inf_nan=False)
    max_error_cap: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    allow_negative_inf: bool = False


def _validate_finite_json(value: Any, path: str = "custom_correctness_kwargs") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _validate_finite_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value")


class Workload(BaseModel):
    """Concrete Kernel Factory-compatible workload configuration."""

    axes: dict[str, Annotated[int, Field(ge=0)] | DynamicAxis]
    inputs: dict[str, InputSpec]
    uuid: str = Field(min_length=1)
    tolerance: ToleranceSpec = Field(default_factory=ToleranceSpec)
    custom_correctness_kwargs: dict[str, Any] = Field(default_factory=dict)
    eval_mode: EvalMode = EvalMode.FULL
    weight: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @field_validator("axes")
    @classmethod
    def _validate_axis_names(cls, axes: dict[str, int | DynamicAxis]) -> dict[str, int | DynamicAxis]:
        if any(not name for name in axes):
            raise ValueError("workload axis names must be non-empty")
        return axes

    @field_validator("custom_correctness_kwargs")
    @classmethod
    def _validate_custom_correctness_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_finite_json(value)
        return value

    @model_validator(mode="after")
    def _validate_cross_field_contract(self) -> "Workload":
        has_dynamic_axes = any(isinstance(value, DynamicAxis) for value in self.axes.values())
        if has_dynamic_axes and self.eval_mode is not EvalMode.CORRECTNESS_ONLY:
            raise ValueError("dynamic axes require eval_mode='correctness_only'")
        custom_count = sum(isinstance(value, CustomInput) for value in self.inputs.values())
        if custom_count and custom_count != len(self.inputs):
            raise ValueError("custom inputs cannot be mixed with other input descriptor types")
        return self
