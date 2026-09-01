# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Loading and Definition-aware validation for inventory Workload JSONL."""

from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator
from uuid import UUID

from pydantic import ValidationError

from tilegym.kernel_inventory.layout import definition_solution_paths_for_workload
from tilegym.kernel_inventory.layout import inventory_coordinate
from tilegym.kernel_inventory.layout import iter_inventory_json_paths
from tilegym.kernel_inventory.layout import iter_inventory_workload_paths
from tilegym.kernel_inventory.layout import mirrored_definition_path
from tilegym.kernel_inventory.layout import mirrored_workload_path
from tilegym.kernel_inventory.schema import CustomInput
from tilegym.kernel_inventory.schema import NullInput
from tilegym.kernel_inventory.schema import RandomInput
from tilegym.kernel_inventory.schema import SafetensorsInput
from tilegym.kernel_inventory.schema import ScalarInput
from tilegym.kernel_inventory.schema import StringInput
from tilegym.kernel_inventory.schema import Workload
from tilegym.kernel_inventory.schema import workload_model_validate

_WORKLOAD_FIELDS = {
    "axes",
    "inputs",
    "uuid",
    "tolerance",
    "custom_correctness_kwargs",
    "eval_mode",
    "weight",
}
_REQUIRED_EXPLICIT_FIELDS = {"axes", "inputs", "uuid", "tolerance", "eval_mode"}
_TOLERANCE_FIELDS = {
    "max_atol",
    "max_rtol",
    "required_matched_ratio",
    "max_error_cap",
    "allow_negative_inf",
}
_EXPECTED_TOLERANCE = {
    "max_atol": 0.02,
    "max_rtol": 0.02,
    "required_matched_ratio": 1.0,
    "max_error_cap": None,
    "allow_negative_inf": False,
}


class KernelWorkloadError(ValueError):
    """Raised when checked-in Workload metadata is invalid."""


@dataclass(frozen=True)
class WorkloadRecord:
    """One parsed Workload plus its physical JSONL source coordinate."""

    path: Path
    line_number: int
    workload: Workload

    @property
    def source_label(self) -> str:
        """Return a path-and-line label for diagnostics."""
        return f"{self.path}:{self.line_number}"


def load_workload_jsonl(path: str | Path) -> tuple[WorkloadRecord, ...]:
    """Load a strict one-object-per-line Workload JSONL file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise KernelWorkloadError(f"{path}: unable to read Workload JSONL: {exc}") from exc
    if not text:
        raise KernelWorkloadError(f"{path}: Workload JSONL must not be empty")

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        label = f"{path}:{line_number}"
        if not line.strip():
            raise KernelWorkloadError(f"{label}: blank lines are not allowed")
        if line.lstrip().startswith("#"):
            raise KernelWorkloadError(f"{label}: comments are not allowed")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KernelWorkloadError(f"{label}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise KernelWorkloadError(f"{label}: expected one JSON object")
        _validate_checked_in_policy(payload, label)
        try:
            workload = workload_model_validate(payload)
        except ValidationError as exc:
            raise KernelWorkloadError(f"{label}: Workload schema invalid: {exc}") from exc
        records.append(WorkloadRecord(path=path, line_number=line_number, workload=workload))

    if not records:
        raise KernelWorkloadError(f"{path}: Workload JSONL must contain at least one row")
    return tuple(records)


def validate_workload_against_definition(
    record: WorkloadRecord,
    definition: dict[str, Any],
    definition_path: str | Path,
) -> None:
    """Validate one Workload row against its adjacent Definition contract."""
    label = record.source_label
    axes = _require_mapping(definition.get("axes"), f"{definition_path}: Definition.axes")
    inputs = _require_mapping(definition.get("inputs"), f"{definition_path}: Definition.inputs")
    variable_axes = {name for name, spec in axes.items() if _require_mapping(spec, f"axis {name}").get("type") == "var"}
    workload_axes = set(record.workload.axes)
    if workload_axes != variable_axes:
        missing = sorted(variable_axes - workload_axes)
        unknown = sorted(workload_axes - variable_axes)
        raise KernelWorkloadError(f"{label}: Workload axes mismatch: missing={missing}, unknown_or_constant={unknown}")
    if any(type(value) is not int or value < 0 for value in record.workload.axes.values()):
        raise KernelWorkloadError(f"{label}: full-evaluation Workload axes must be concrete non-negative integers")

    workload_inputs = set(record.workload.inputs)
    definition_inputs = set(inputs)
    if workload_inputs != definition_inputs:
        missing = sorted(definition_inputs - workload_inputs)
        unknown = sorted(workload_inputs - definition_inputs)
        raise KernelWorkloadError(f"{label}: Workload inputs mismatch: missing={missing}, unknown={unknown}")

    none_defaults = _reference_none_defaults(definition.get("reference", ""))
    for name, descriptor in record.workload.inputs.items():
        spec = _require_mapping(inputs[name], f"{definition_path}: Definition.inputs.{name}")
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if isinstance(descriptor, RandomInput | SafetensorsInput):
            if shape is None:
                raise KernelWorkloadError(f"{label}: input {name!r} uses a tensor descriptor for scalar TensorSpec")
            if isinstance(descriptor, SafetensorsInput):
                locators = descriptor.shards if descriptor.shards is not None else (descriptor,)
                for locator in locators:
                    _resolve_safetensors_path(record, locator.path)
        elif isinstance(descriptor, ScalarInput):
            if shape is not None:
                raise KernelWorkloadError(f"{label}: input {name!r} uses a scalar descriptor for tensor TensorSpec")
            _validate_scalar_dtype(descriptor.value, dtype, f"{label}: input {name!r}")
        elif isinstance(descriptor, NullInput):
            if name not in none_defaults:
                raise KernelWorkloadError(
                    f"{label}: null input {name!r} requires an explicit None default in Definition.reference.run"
                )
        elif isinstance(descriptor, StringInput):
            if shape is not None:
                raise KernelWorkloadError(f"{label}: input {name!r} uses a string descriptor for tensor TensorSpec")
        elif isinstance(descriptor, CustomInput):
            if not definition.get("custom_inputs_entrypoint"):
                raise KernelWorkloadError(f"{label}: custom inputs require Definition.custom_inputs_entrypoint")

    _validate_resolvable_constraints(record, definition, axes)


def validate_workload_catalog(root: str | Path, *, require_complete: bool) -> None:
    """Validate active Workload topology, rows, UUIDs, and adjacent contracts."""
    root = Path(root)
    definition_paths = set(iter_inventory_json_paths(root, "kernel_definitions"))
    workload_paths = set(iter_inventory_workload_paths(root))
    definition_by_workload = {mirrored_workload_path(path): path for path in definition_paths}

    stale = sorted(workload_paths - set(definition_by_workload))
    if stale:
        raise KernelWorkloadError(f"Stale Workload files without active Definitions: {stale}")
    if require_complete:
        missing = sorted(set(definition_by_workload) - workload_paths)
        if missing:
            raise KernelWorkloadError(f"Active Definitions missing mirrored Workloads: {missing}")

    uuid_sources: dict[str, str] = {}
    asset_owners: dict[Path, set[Path]] = {}
    for workload_path in sorted(workload_paths):
        definition_path = mirrored_definition_path(workload_path)
        definition = _load_json_object(definition_path)
        targets = list(definition_solution_paths_for_workload(workload_path))
        if not targets:
            raise KernelWorkloadError(f"{workload_path}: adjacent Definition has no accepted Solution target")
        for record in load_workload_jsonl(workload_path):
            previous = uuid_sources.get(record.workload.uuid)
            if previous is not None:
                raise KernelWorkloadError(
                    f"{record.source_label}: duplicate Workload UUID {record.workload.uuid!r}; first seen at {previous}"
                )
            uuid_sources[record.workload.uuid] = record.source_label
            validate_workload_against_definition(record, definition, definition_path)
            for asset_path in _safetensors_paths(record):
                asset_owners.setdefault(asset_path, set()).add(workload_path)

    shared_assets = {path: owners for path, owners in asset_owners.items() if len(owners) > 1}
    if shared_assets:
        details = {str(path): [str(owner) for owner in sorted(owners)] for path, owners in shared_assets.items()}
        raise KernelWorkloadError(f"Safetensors assets must have exactly one owning Workload: {details}")

    inventory_roots = {inventory_coordinate(path).inventory_root for path in definition_paths}
    checked_in_assets = {
        path.resolve()
        for inventory_root in inventory_roots
        for path in (inventory_root / "kernel_workloads").rglob("*.safetensors")
    }
    orphan_assets = sorted(checked_in_assets - set(asset_owners))
    if orphan_assets:
        raise KernelWorkloadError(f"Orphan safetensors files without Workload references: {orphan_assets}")


def iter_workload_records(root: str | Path) -> Iterator[WorkloadRecord]:
    """Yield every parsed Workload row in deterministic path/line order."""
    for path in iter_inventory_workload_paths(root):
        yield from load_workload_jsonl(path)


def materialize_workload_inputs(
    record: WorkloadRecord,
    definition: dict[str, Any],
    *,
    torch: Any,
    device: Any,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Materialize concrete axes and inputs for one validated Workload row."""
    definition_axes = _require_mapping(definition.get("axes"), "Definition.axes")
    axes = {
        name: spec["value"]
        for name, spec in definition_axes.items()
        if isinstance(spec, dict) and spec.get("type") == "const"
    }
    axes.update({name: int(value) for name, value in record.workload.axes.items()})

    definition_inputs = _require_mapping(definition.get("inputs"), "Definition.inputs")
    materialized = {}
    for name, descriptor in record.workload.inputs.items():
        spec = _require_mapping(definition_inputs[name], f"Definition.inputs.{name}")
        if isinstance(descriptor, RandomInput):
            materialized[name] = _materialize_random_tensor(spec, axes, torch, device)
        elif isinstance(descriptor, SafetensorsInput):
            materialized[name] = _materialize_safetensors_tensor(record, descriptor, spec, axes, torch, device)
        elif isinstance(descriptor, ScalarInput | StringInput):
            materialized[name] = descriptor.value
        elif isinstance(descriptor, NullInput):
            materialized[name] = None
        elif isinstance(descriptor, CustomInput):
            raise KernelWorkloadError(
                f"{record.source_label}: TileGym runtime materialization does not yet support custom inputs"
            )
        else:  # pragma: no cover - selected schema union prevents this
            raise KernelWorkloadError(f"{record.source_label}: unsupported input descriptor for {name!r}")
    return axes, materialized


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
        raise KernelWorkloadError(f"Unsupported inventory dtype: {dtype}")
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise KernelWorkloadError(f"PyTorch does not provide inventory dtype: {dtype}") from exc


def _materialize_random_tensor(spec: dict[str, Any], axes: dict[str, int], torch: Any, device: Any) -> Any:
    shape = _resolved_shape(spec, axes)
    dtype_name = spec["dtype"]
    dtype = resolve_torch_dtype(dtype_name, torch)
    if dtype_name == "bool":
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype_name.startswith("uint"):
        return torch.randint(0, 4, shape, dtype=torch.int64, device=device).to(dtype)
    if dtype_name.startswith("int"):
        return torch.randint(-4, 4, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)


def _materialize_safetensors_tensor(
    record: WorkloadRecord,
    descriptor: SafetensorsInput,
    spec: dict[str, Any],
    axes: dict[str, int],
    torch: Any,
    device: Any,
) -> Any:
    if descriptor.shards is not None:
        rank = int(os.environ.get("RANK", "0"))
        if rank < 0 or rank >= len(descriptor.shards):
            raise KernelWorkloadError(
                f"{record.source_label}: rank {rank} has no safetensors shard among {len(descriptor.shards)} locators"
            )
        locator = descriptor.shards[rank]
        raw_path, tensor_key = locator.path, locator.tensor_key
    else:
        assert descriptor.path is not None and descriptor.tensor_key is not None
        raw_path, tensor_key = descriptor.path, descriptor.tensor_key

    tensor_path = _resolve_safetensors_path(record, raw_path)

    try:
        from safetensors.torch import load_file

        values = load_file(tensor_path.as_posix(), device="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise KernelWorkloadError(f"{record.source_label}: unable to load safetensors file {raw_path}: {exc}") from exc
    if tensor_key not in values:
        raise KernelWorkloadError(f"{record.source_label}: safetensors key {tensor_key!r} not found in {raw_path}")
    value = values[tensor_key]
    expected_shape = _resolved_shape(spec, axes)
    expected_dtype = resolve_torch_dtype(spec["dtype"], torch)
    if tuple(value.shape) != expected_shape:
        raise KernelWorkloadError(
            f"{record.source_label}: safetensors input shape {tuple(value.shape)} does not match {expected_shape}"
        )
    if value.dtype != expected_dtype:
        raise KernelWorkloadError(
            f"{record.source_label}: safetensors input dtype {value.dtype} does not match {expected_dtype}"
        )
    return value.to(device=device).contiguous()


def _resolve_safetensors_path(record: WorkloadRecord, raw_path: str) -> Path:
    owning_directory = record.path.parent.resolve()
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise KernelWorkloadError(f"{record.source_label}: safetensors path must be relative: {raw_path}")
    if raw_path != relative_path.name or relative_path.suffix != ".safetensors":
        raise KernelWorkloadError(
            f"{record.source_label}: safetensors path must be a .safetensors basename beside workload.jsonl: {raw_path}"
        )
    tensor_path = (owning_directory / relative_path).resolve()
    try:
        tensor_path.relative_to(owning_directory)
    except ValueError as exc:
        raise KernelWorkloadError(
            f"{record.source_label}: safetensors path escapes Workload directory: {raw_path}"
        ) from exc
    if not tensor_path.is_file():
        raise KernelWorkloadError(f"{record.source_label}: safetensors file does not exist: {raw_path}")
    return tensor_path


def _safetensors_paths(record: WorkloadRecord) -> Iterator[Path]:
    for descriptor in record.workload.inputs.values():
        if not isinstance(descriptor, SafetensorsInput):
            continue
        locators = descriptor.shards if descriptor.shards is not None else (descriptor,)
        for locator in locators:
            yield _resolve_safetensors_path(record, locator.path)


def _resolved_shape(spec: dict[str, Any], axes: dict[str, int]) -> tuple[int, ...]:
    shape = spec.get("shape")
    if not isinstance(shape, list):
        raise KernelWorkloadError(f"Tensor input requires a list shape, got {shape!r}")
    try:
        return tuple(axes[dimension] for dimension in shape)
    except KeyError as exc:
        raise KernelWorkloadError(f"Tensor shape references unresolved axis {exc.args[0]!r}") from exc


def _validate_checked_in_policy(payload: dict[str, Any], label: str) -> None:
    _validate_finite_json(payload, label)
    non_string_fields = [key for key in payload if not isinstance(key, str)]
    if non_string_fields:
        raise KernelWorkloadError(f"{label}: Workload field names must be strings: {non_string_fields!r}")
    unknown = sorted(set(payload) - _WORKLOAD_FIELDS)
    if unknown:
        raise KernelWorkloadError(f"{label}: unsupported Workload fields: {unknown}")
    missing = sorted(_REQUIRED_EXPLICIT_FIELDS - set(payload))
    if missing:
        raise KernelWorkloadError(f"{label}: missing explicit Workload fields: {missing}")

    tolerance = _require_mapping(payload["tolerance"], f"{label}: tolerance")
    tolerance_fields = set(tolerance)
    if tolerance_fields != _TOLERANCE_FIELDS:
        missing_tolerance = sorted(_TOLERANCE_FIELDS - tolerance_fields)
        unknown_tolerance = sorted(tolerance_fields - _TOLERANCE_FIELDS)
        raise KernelWorkloadError(
            f"{label}: tolerance fields mismatch: missing={missing_tolerance}, unknown={unknown_tolerance}"
        )
    if tolerance != _EXPECTED_TOLERANCE:
        raise KernelWorkloadError(
            f"{label}: tolerance must equal the checked-in Workload migration policy {_EXPECTED_TOLERANCE}"
        )
    if payload["eval_mode"] != "full":
        raise KernelWorkloadError(f"{label}: eval_mode must be 'full'")

    try:
        uuid = UUID(payload.get("uuid", ""))
    except (AttributeError, TypeError, ValueError) as exc:
        raise KernelWorkloadError(f"{label}: uuid must be an RFC 4122 UUIDv4 string") from exc
    if uuid.version != 4 or str(uuid) != payload["uuid"].lower():
        raise KernelWorkloadError(f"{label}: uuid must be a canonical RFC 4122 UUIDv4 string")


def _validate_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise KernelWorkloadError(f"{label}: Workload JSON numbers must be finite")
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item, label)


def _validate_scalar_dtype(value: Any, dtype: Any, label: str) -> None:
    if dtype == "bool":
        valid = type(value) is bool
    elif isinstance(dtype, str) and dtype.startswith(("int", "uint")):
        valid = type(value) is int
    elif isinstance(dtype, str) and dtype.startswith(("float", "bfloat")):
        valid = type(value) in {int, float}
    else:
        valid = isinstance(value, str) if dtype in {"str", "string"} else False
    if not valid:
        raise KernelWorkloadError(f"{label} value {value!r} is incompatible with scalar dtype {dtype!r}")


def _reference_none_defaults(reference: Any) -> set[str]:
    if not isinstance(reference, str):
        return set()
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return set()
    run = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"), None)
    if run is None:
        return set()
    positional = [*run.args.posonlyargs, *run.args.args]
    positional_defaults = [None] * (len(positional) - len(run.args.defaults)) + list(run.args.defaults)
    defaults = {argument.arg: default for argument, default in zip(positional, positional_defaults, strict=True)}
    defaults.update(dict(zip((argument.arg for argument in run.args.kwonlyargs), run.args.kw_defaults, strict=True)))
    return {name for name, default in defaults.items() if isinstance(default, ast.Constant) and default.value is None}


def _validate_resolvable_constraints(
    record: WorkloadRecord,
    definition: dict[str, Any],
    definition_axes: dict[str, Any],
) -> None:
    context = {
        name: spec["value"]
        for name, spec in definition_axes.items()
        if isinstance(spec, dict) and spec.get("type") == "const"
    }
    context.update(record.workload.axes)
    context.update(
        {
            name: descriptor.value
            for name, descriptor in record.workload.inputs.items()
            if isinstance(descriptor, ScalarInput | StringInput)
        }
    )
    context.update(
        {name: None for name, descriptor in record.workload.inputs.items() if isinstance(descriptor, NullInput)}
    )
    for constraint in definition.get("constraints", []):
        if not isinstance(constraint, str):
            continue
        try:
            expression = ast.parse(constraint, mode="eval")
        except SyntaxError:
            continue
        names = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
        if not names <= set(context):
            continue
        try:
            holds = eval(compile(expression, "<Definition.constraints>", "eval"), {"__builtins__": {}}, context)
        except (NameError, TypeError, ValueError):
            continue
        if not holds:
            raise KernelWorkloadError(f"{record.source_label}: Workload violates Definition constraint: {constraint}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KernelWorkloadError(f"{label} must be an object")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelWorkloadError(f"{path}: unable to load Definition: {exc}") from exc
    return _require_mapping(value, str(path))


__all__ = [
    "KernelWorkloadError",
    "WorkloadRecord",
    "iter_workload_records",
    "load_workload_jsonl",
    "materialize_workload_inputs",
    "resolve_torch_dtype",
    "validate_workload_against_definition",
    "validate_workload_catalog",
]
