# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Declarative runtime-input patterns for kernel inventory correctness tests."""

from __future__ import annotations

import ast
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory.layout import inventory_coordinate
    from tilegym.kernel_inventory.layout import iter_inventory_json_paths
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)


RUNTIME_INPUTS_PATH = Path(__file__).with_name("runtime_inputs.yaml")
_GENERATOR_KINDS = {"arange", "default", "full", "none", "normal", "ones", "randint", "values", "zeros"}
_GENERATOR_FIELDS = {
    "arange": {"kind", "start", "step"},
    "default": {"kind"},
    "full": {"kind", "value"},
    "none": {"kind"},
    "normal": {"kind", "mean", "std"},
    "ones": {"kind"},
    "randint": {"kind", "low", "high"},
    "values": {"kind", "value"},
    "zeros": {"kind"},
}
_GENERATOR_REQUIRED_FIELDS = {
    "full": {"value"},
    "randint": {"high"},
    "values": {"value"},
}
_INTEGER_DTYPE_BOUNDS = {
    "int8": (-(2**7), 2**7 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
    "uint8": (0, 2**8 - 1),
    "uint16": (0, 2**16 - 1),
    "uint32": (0, 2**32 - 1),
    "uint64": (0, 2**64 - 1),
}


@dataclass(frozen=True)
class RuntimeInputCase:
    """Merged axis and input overrides for one canonical Definition."""

    axes: dict[str, int]
    inputs: dict[str, dict[str, Any]]
    mutated_inputs: tuple[str, ...]


class RuntimeInputCatalog:
    """Validated reusable input patterns loaded from YAML."""

    def __init__(self, data: dict[str, Any]):
        non_string_fields = [key for key in data if not isinstance(key, str)]
        if non_string_fields:
            raise ValueError(f"runtime input catalog field names must be strings: {non_string_fields!r}")
        unknown = sorted(key for key in data if key not in {"version", "patterns", "cases"})
        if unknown:
            raise ValueError(f"runtime input catalog has unsupported fields: {unknown}")
        version = data.get("version")
        if type(version) is not int or version != 1:
            raise ValueError("runtime input catalog version must be 1")
        self.patterns = _mapping(data.get("patterns", {}), "patterns")
        self.cases = _mapping(data.get("cases", {}), "cases")
        for name, pattern in self.patterns.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"runtime input pattern names must be nonempty strings: {name!r}")
            self.patterns[name] = _validate_case(pattern, f"patterns.{name}", allow_patterns=False)
        for canonical_id, case in self.cases.items():
            if not isinstance(canonical_id, str) or "::definition::" not in canonical_id:
                raise ValueError(f"Invalid canonical Definition id in runtime input catalog: {canonical_id!r}")
            self.cases[canonical_id] = _validate_case(case, f"cases.{canonical_id}", allow_patterns=True)
            missing_patterns = sorted(set(self.cases[canonical_id]["patterns"]) - set(self.patterns))
            if missing_patterns:
                raise ValueError(f"cases.{canonical_id} references missing patterns: {missing_patterns}")

    @classmethod
    def from_path(
        cls,
        path: str | Path = RUNTIME_INPUTS_PATH,
        *,
        definition_paths: Iterable[str | Path] | None = None,
    ) -> "RuntimeInputCatalog":
        data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        catalog = cls(_mapping(data, "runtime input catalog"))
        if definition_paths is None and Path(path).resolve() == RUNTIME_INPUTS_PATH.resolve():
            definition_paths = iter_inventory_json_paths(REPO_ROOT, "kernel_definitions")
        if definition_paths is not None:
            catalog.validate_definition_ids(definition_paths)
        return catalog

    def validate_definition_ids(self, definition_paths: Iterable[str | Path]) -> None:
        """Reject stale or schema-incompatible cases before runtime dependency gates."""
        discovered: dict[str, tuple[Path, dict[str, Any]]] = {}
        for raw_path in definition_paths:
            path = Path(raw_path)
            canonical_id = inventory_coordinate(path).canonical_id
            if canonical_id in discovered:
                raise ValueError(f"duplicate canonical Definition path in runtime input validation: {canonical_id}")
            definition = json.loads(path.read_text(encoding="utf-8"))
            discovered[canonical_id] = (path, _mapping(definition, str(path)))
        stale = sorted(set(self.cases) - set(discovered))
        if stale:
            raise ValueError(f"runtime input catalog contains stale canonical Definition ids: {stale}")
        for canonical_id in self.cases:
            path, definition = discovered[canonical_id]
            _validate_merged_case(self.case_for_definition(path), definition, canonical_id)

    def case_for_definition(self, definition_path: str | Path) -> RuntimeInputCase:
        canonical_id = inventory_coordinate(definition_path).canonical_id
        case = self.cases.get(canonical_id, {"patterns": [], "axes": {}, "inputs": {}, "mutates": []})
        axes: dict[str, int] = {}
        inputs: dict[str, dict[str, Any]] = {}
        mutated_inputs: list[str] = []
        for pattern_name in case["patterns"]:
            pattern = self.patterns[pattern_name]
            axes.update(pattern["axes"])
            inputs.update(pattern["inputs"])
            mutated_inputs.extend(pattern["mutates"])
        axes.update(case["axes"])
        inputs.update(case["inputs"])
        mutated_inputs.extend(case["mutates"])
        return RuntimeInputCase(axes=axes, inputs=inputs, mutated_inputs=tuple(dict.fromkeys(mutated_inputs)))


def make_runtime_override(
    config: dict[str, Any],
    spec: dict[str, Any],
    axes: dict[str, int],
    torch: Any,
    device: Any,
) -> Any:
    """Materialize one override while deriving shape and dtype from Definition."""
    kind = config["kind"]
    if kind in {"default", "none"}:
        return None
    if spec["shape"] is None:
        if kind != "full":
            raise ValueError(f"Scalar runtime inputs require kind=full, got {kind}")
        return config.get("value")

    shape = tuple(axes[axis] for axis in spec["shape"])
    dtype = resolve_torch_dtype(spec["dtype"], torch)
    if kind == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if kind == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    if kind == "full":
        return torch.full(shape, config["value"], dtype=dtype, device=device)
    if kind == "normal":
        mean = float(config.get("mean", 0.0))
        std = float(config.get("std", 1.0))
        return (mean + std * torch.randn(shape, dtype=torch.float32, device=device)).to(dtype)
    if kind == "randint":
        generated_dtype = torch.int64 if spec["dtype"].startswith("uint") else dtype
        return torch.randint(
            int(config.get("low", 0)), int(config["high"]), shape, dtype=generated_dtype, device=device
        ).to(dtype)
    if kind == "arange":
        start = config.get("start", 0)
        step = config.get("step", 1)
        count = 1
        for size in shape:
            count *= size
        generated_dtype = torch.int64 if spec["dtype"].startswith("uint") else dtype
        return (start + step * torch.arange(count, dtype=generated_dtype, device=device)).to(dtype).reshape(shape)
    if kind == "values":
        value = torch.tensor(config["value"], dtype=dtype, device=device)
        if tuple(value.shape) != shape:
            raise ValueError(f"Literal runtime input has shape {tuple(value.shape)}, expected {shape}")
        return value
    raise ValueError(f"Unsupported runtime input generator: {kind}")


def _validate_case(value: Any, label: str, *, allow_patterns: bool) -> dict[str, Any]:
    value = _mapping(value, label)
    allowed = {"axes", "inputs", "mutates"} | ({"patterns"} if allow_patterns else set())
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {unknown}")
    patterns = value.get("patterns", [])
    if not isinstance(patterns, list) or not all(isinstance(name, str) and name for name in patterns):
        raise ValueError(f"{label}.patterns must be a list of names")
    axes = _mapping(value.get("axes", {}), f"{label}.axes")
    if not all(isinstance(name, str) and type(size) is int and size > 0 for name, size in axes.items()):
        raise ValueError(f"{label}.axes values must be positive integers")
    inputs = _mapping(value.get("inputs", {}), f"{label}.inputs")
    mutates = value.get("mutates", [])
    if (
        not isinstance(mutates, list)
        or not all(isinstance(name, str) and name for name in mutates)
        or len(set(mutates)) != len(mutates)
    ):
        raise ValueError(f"{label}.mutates must be a list of unique input names")
    normalized_inputs = {}
    for name, config in inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.inputs keys must be nonempty input names")
        config = _mapping(config, f"{label}.inputs.{name}")
        kind = config.get("kind")
        if not isinstance(kind, str) or kind not in _GENERATOR_KINDS:
            raise ValueError(f"{label}.inputs.{name}.kind must be one of {sorted(_GENERATOR_KINDS)}")
        non_string_fields = [field for field in config if not isinstance(field, str)]
        if non_string_fields:
            raise ValueError(f"{label}.inputs.{name} generator field names must be strings: {non_string_fields!r}")
        unknown_generator_fields = sorted(set(config) - _GENERATOR_FIELDS[kind])
        if unknown_generator_fields:
            raise ValueError(
                f"{label}.inputs.{name} {kind} generator has unsupported fields: {unknown_generator_fields}"
            )
        missing_generator_fields = sorted(_GENERATOR_REQUIRED_FIELDS.get(kind, set()) - set(config))
        if missing_generator_fields:
            raise ValueError(f"{label}.inputs.{name} {kind} generator requires fields: {missing_generator_fields}")
        _validate_generator_parameters(config, f"{label}.inputs.{name}")
        normalized_inputs[name] = config
    return {"patterns": patterns, "axes": axes, "inputs": normalized_inputs, "mutates": mutates}


def _validate_merged_case(case: RuntimeInputCase, definition: dict[str, Any], canonical_id: str) -> None:
    axes = _mapping(definition.get("axes", {}), f"{canonical_id}.axes")
    inputs = _mapping(definition.get("inputs", {}), f"{canonical_id}.inputs")
    unknown_axes = sorted(set(case.axes) - set(axes))
    if unknown_axes:
        raise ValueError(f"{canonical_id} runtime case overrides unknown axes: {unknown_axes}")
    conflicting_constants = sorted(
        name
        for name, value in case.axes.items()
        if axes[name].get("type") == "const" and axes[name].get("value") != value
    )
    if conflicting_constants:
        raise ValueError(f"{canonical_id} runtime case contradicts const axes: {conflicting_constants}")
    unknown_inputs = sorted(set(case.inputs) - set(inputs))
    if unknown_inputs:
        raise ValueError(f"{canonical_id} runtime case overrides unknown inputs: {unknown_inputs}")
    unknown_mutations = sorted(set(case.mutated_inputs) - set(inputs))
    if unknown_mutations:
        raise ValueError(f"{canonical_id} runtime case mutates unknown inputs: {unknown_mutations}")
    none_overrides = {name for name, config in case.inputs.items() if config["kind"] == "none"}
    invalid_none_overrides = (
        sorted(none_overrides - _reference_none_defaults(definition, canonical_id)) if none_overrides else []
    )
    if invalid_none_overrides:
        raise ValueError(
            f"{canonical_id} none overrides require an explicit reference.run default of None: {invalid_none_overrides}"
        )
    incompatible_scalars = sorted(
        name
        for name, config in case.inputs.items()
        if inputs[name].get("shape") is None and config["kind"] not in {"default", "full", "none"}
    )
    if incompatible_scalars:
        raise ValueError(
            f"{canonical_id} scalar runtime inputs require default/full generators: {incompatible_scalars}"
        )
    for name, config in case.inputs.items():
        _validate_generator_for_spec(config, inputs[name], f"{canonical_id}.inputs.{name}")


def _validate_generator_parameters(config: dict[str, Any], label: str) -> None:
    kind = config["kind"]
    if kind == "normal":
        mean = config.get("mean", 0.0)
        std = config.get("std", 1.0)
        if not _is_finite_number(mean) or not _is_finite_number(std) or std < 0:
            raise ValueError(f"{label} normal mean/std must be finite numbers with nonnegative std")
    elif kind == "arange":
        start = config.get("start", 0)
        step = config.get("step", 1)
        if not _is_finite_number(start) or not _is_finite_number(step) or step == 0:
            raise ValueError(f"{label} arange start/step must be finite numbers with nonzero step")
    elif kind == "randint":
        low = config.get("low", 0)
        high = config["high"]
        if not _is_integer(low) or not _is_integer(high) or high <= low:
            raise ValueError(f"{label} randint low/high must be integers with high greater than low")


def _validate_generator_for_spec(config: dict[str, Any], spec: dict[str, Any], label: str) -> None:
    kind = config["kind"]
    dtype = spec.get("dtype")
    is_scalar = spec.get("shape") is None
    if kind == "normal" and not _is_floating_dtype(dtype):
        raise ValueError(f"{label} normal generator requires a floating-point Definition dtype")
    if kind == "randint" and dtype not in _INTEGER_DTYPE_BOUNDS:
        raise ValueError(f"{label} randint generator requires an integer Definition dtype")
    if kind == "randint" and dtype in _INTEGER_DTYPE_BOUNDS:
        low, high = config.get("low", 0), config["high"]
        dtype_low, dtype_high = _INTEGER_DTYPE_BOUNDS[dtype]
        if low < dtype_low or high - 1 > dtype_high:
            raise ValueError(f"{label} randint bounds exceed Definition dtype {dtype}")
    if kind == "arange" and dtype == "bool":
        raise ValueError(f"{label} arange generator does not support Boolean Definition dtype")
    if kind == "arange" and dtype in _INTEGER_DTYPE_BOUNDS:
        if not _is_integer(config.get("start", 0)) or not _is_integer(config.get("step", 1)):
            raise ValueError(f"{label} integer arange parameters must be integers")
    if kind == "full" and not _dtype_value_is_valid(config["value"], dtype):
        level = "scalar " if is_scalar else ""
        raise ValueError(f"{label} full value is incompatible with {level}Definition dtype {dtype}")
    if kind == "values":
        value = config["value"]
        if not isinstance(value, list) or not _nested_values_match_dtype(value, dtype):
            raise ValueError(f"{label} literal values are incompatible with Definition dtype {dtype}")


def _reference_none_defaults(definition: dict[str, Any], label: str) -> set[str]:
    try:
        tree = ast.parse(definition.get("reference", ""), filename=label)
    except (SyntaxError, TypeError) as exc:
        raise ValueError(f"{label} Definition.reference must be valid Python") from exc
    runs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
    ]
    if len(runs) != 1:
        raise ValueError(f"{label} Definition.reference must define exactly one global run function")
    run = runs[0]
    defaults: dict[str, ast.expr] = {}
    positional = [*run.args.posonlyargs, *run.args.args]
    if run.args.defaults:
        defaults.update(
            (parameter.arg, value)
            for parameter, value in zip(positional[-len(run.args.defaults) :], run.args.defaults, strict=True)
        )
    defaults.update(
        (parameter.arg, value)
        for parameter, value in zip(run.args.kwonlyargs, run.args.kw_defaults, strict=True)
        if value is not None
    )
    return {name for name, value in defaults.items() if isinstance(value, ast.Constant) and value.value is None}


def _nested_values_match_dtype(value: list[Any], dtype: Any) -> bool:
    return all(
        _nested_values_match_dtype(item, dtype) if isinstance(item, list) else _dtype_value_is_valid(item, dtype)
        for item in value
    )


def _dtype_value_is_valid(value: Any, dtype: Any) -> bool:
    if dtype == "bool":
        return type(value) is bool
    if dtype in _INTEGER_DTYPE_BOUNDS:
        if not _is_integer(value):
            return False
        low, high = _INTEGER_DTYPE_BOUNDS[dtype]
        return low <= value <= high
    if _is_floating_dtype(dtype):
        return _is_finite_number(value)
    return False


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


def _is_floating_dtype(dtype: Any) -> bool:
    return isinstance(dtype, str) and (dtype.startswith("float") or dtype == "bfloat16")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses silent mapping-key replacement."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"runtime input catalog contains duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def resolve_torch_dtype(dtype: str, torch: Any) -> Any:
    """Resolve an inventory dtype string to its torch dtype."""
    supported = {
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
        "bool",
    }
    if dtype not in supported:
        raise ValueError(f"Unsupported inventory dtype: {dtype}")
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"PyTorch does not provide inventory dtype: {dtype}") from exc
