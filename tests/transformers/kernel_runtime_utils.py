# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import inventory modules without running tilegym/__init__.py. Runtime tests
# import the individual solution modules directly from Solution.spec.entry_point.
tilegym_pkg = types.ModuleType("tilegym")
tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
sys.modules.setdefault("tilegym", tilegym_pkg)
transformers_pkg = types.ModuleType("tilegym.transformers")
transformers_pkg.__path__ = [str(REPO_ROOT / "src/tilegym/transformers")]
sys.modules.setdefault("tilegym.transformers", transformers_pkg)

from tilegym.transformers.kernel_inventory import iter_kernel_definition_paths
from tilegym.transformers.kernel_inventory import load_json
from tilegym.transformers.kernel_inventory import validate_definition
from tilegym.transformers.kernel_inventory import validate_solution
from tilegym.transformers.kernel_inventory import validate_solution_entry_point

DEFAULT_AXIS_VALUES = {
    "N": 3,
    "D": 64,
    "H": 64,
    "T": 5,
}
MUTATED_INPUTS_BY_DEFINITION = {
    "olmo3_dual_rms_norm": ("q", "k"),
    "olmoe_dual_rms_norm": ("q", "k"),
    "qwen3_5_causal_conv1d_update_silu": ("conv_state",),
}


def all_definition_solution_cases() -> list[Any]:
    return definition_solution_cases_for_submodule(None)


def definition_solution_cases_for_submodule(submodule: str | Path | None) -> list[Any]:
    search_root = _transformer_submodule_root(submodule)
    cases = []
    definition_paths = (
        iter_kernel_definition_paths(REPO_ROOT)
        if search_root == REPO_ROOT
        else sorted(search_root.glob("kernel_definitions/*.json"))
    )
    for definition_path in definition_paths:
        solution_path = definition_path.parent.parent / "kernel_solutions" / definition_path.name
        definition_name = load_json(definition_path)["name"]
        cases.append(pytest.param(definition_path, solution_path, id=definition_name))
    if search_root != REPO_ROOT and not cases:
        raise ValueError(f"Transformer submodule has no kernel Definitions: {search_root}")
    return cases


def _transformer_submodule_root(submodule: str | Path | None) -> Path:
    if submodule is None or str(submodule) == "":
        return REPO_ROOT

    raw_path = Path(submodule)
    if raw_path.is_absolute():
        submodule_root = raw_path
    else:
        for candidate in (
            Path.cwd() / raw_path,
            REPO_ROOT / raw_path,
            REPO_ROOT / "src/tilegym/transformers" / raw_path,
        ):
            if candidate.exists():
                submodule_root = candidate
                break
        else:
            submodule_root = REPO_ROOT / "src/tilegym/transformers" / raw_path

    submodule_root = submodule_root.resolve()
    transformers_root = (REPO_ROOT / "src/tilegym/transformers").resolve()
    try:
        submodule_root.relative_to(transformers_root)
    except ValueError as exc:
        raise ValueError(f"Transformer submodule must live under {transformers_root}: {submodule}") from exc

    if not submodule_root.is_dir():
        raise ValueError(f"Transformer submodule does not exist: {submodule}")
    return submodule_root


def run_definition_solution_runtime(definition_path: Path, solution_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("flashinfer_bench")
    pytest.importorskip("cuda.tile")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for transformer kernel runtime checks")

    definition = load_json(definition_path)
    solution = load_json(solution_path)
    validate_definition(definition)
    validate_solution(solution, repo_root=REPO_ROOT)
    validate_solution_entry_point(solution, repo_root=REPO_ROOT)
    _skip_if_solution_does_not_target_current_hardware(solution, torch)

    device = torch.device("cuda")
    torch.manual_seed(2026)
    inputs = _make_inputs(definition, torch, device)
    reference_inputs = {name: _clone_value(value) for name, value in inputs.items()}
    solution_inputs = {name: _clone_value(value) for name, value in inputs.items()}

    reference_fn = _load_reference(definition["reference"], definition_path)
    reference_outputs = _as_tuple(reference_fn(*(reference_inputs[name] for name in definition["inputs"])))

    solution_fn = _load_solution_entry(solution)
    solution_outputs = _as_tuple(_call_solution(solution_fn, solution_inputs))
    torch.cuda.synchronize()

    output_names = list(definition["outputs"])
    assert len(solution_outputs) == len(reference_outputs) == len(output_names), (
        f"{definition['name']}: output arity mismatch for outputs {output_names}"
    )
    for name, actual, expected in zip(output_names, solution_outputs, reference_outputs):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda msg: f"{definition['name']} output {name} mismatch\n{msg}",
        )

    for name in MUTATED_INPUTS_BY_DEFINITION.get(definition["name"], ()):
        torch.testing.assert_close(
            solution_inputs[name],
            reference_inputs[name],
            rtol=2e-2,
            atol=2e-2,
            msg=lambda msg: f"{definition['name']} mutated input {name} mismatch\n{msg}",
        )


def _axis_values(definition: dict[str, Any]) -> dict[str, int]:
    values = {name: axis["value"] for name, axis in definition["axes"].items() if axis.get("type") == "const"}
    for name, axis in definition["axes"].items():
        if axis.get("type") == "var":
            values[name] = DEFAULT_AXIS_VALUES.get(name, 4)
    if "T_padded" in values and "T" in values and "K" in values:
        values["T_padded"] = values["T"] + values["K"] - 1
    return values


def _make_inputs(definition: dict[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    axes = _axis_values(definition)
    inputs = {}
    for name, spec in definition["inputs"].items():
        inputs[name] = _make_input(name, spec, axes, torch, device)
    return inputs


def _make_input(name: str, spec: dict[str, Any], axes: dict[str, int], torch: Any, device: Any) -> Any:
    dtype = spec["dtype"]
    shape_spec = spec["shape"]
    if shape_spec is None:
        if name == "eps":
            return 1e-6
        if name == "offset":
            return 1.0
        if name == "seq_len":
            return axes["T"]
        if dtype.startswith("int"):
            return 1
        return 1.0

    shape = tuple(axes[axis] for axis in shape_spec)
    torch_dtype = _torch_dtype(dtype, torch)
    if dtype == "bool":
        return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    if dtype.startswith("int"):
        return torch.randint(-3, 4, shape, device=device, dtype=torch_dtype)

    base = 0.1 * torch.randn(shape, device=device, dtype=torch.float32)
    if "weight" in name:
        base = 1.0 + base
    if name == "A_log":
        base = -base.abs()
    return base.to(torch_dtype).contiguous()


def _torch_dtype(dtype: str, torch: Any) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }[dtype]


def _skip_if_solution_does_not_target_current_hardware(solution: dict[str, Any], torch: Any) -> None:
    target_hardware = solution["spec"].get("target_hardware", [])
    current_hardware = _current_hardware_label(torch)
    if target_hardware and current_hardware not in target_hardware:
        pytest.skip(
            f"Solution targets {target_hardware}; current hardware is {current_hardware} "
            f"({torch.cuda.get_device_name(0)})"
        )


def _current_hardware_label(torch: Any) -> str:
    return torch.cuda.get_device_name(0).upper().replace(" ", "_").replace("-", "_")


def _load_reference(reference: str, definition_path: Path) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(reference, str(definition_path), "exec"), namespace)
    return namespace["run"]


def _load_solution_entry(solution: dict[str, Any]) -> Any:
    file_path, symbol = solution["spec"]["entry_point"].split("::", 1)
    source_path = REPO_ROOT / file_path
    module_name = f"_tilegym_transformer_runtime_{source_path.stem}_{abs(hash(source_path))}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _call_solution(solution_fn: Any, inputs: dict[str, Any]) -> Any:
    signature = inspect.signature(solution_fn)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in inputs:
            kwargs[name] = inputs[name]
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(f"Solution entry point requires argument not present in Definition.inputs: {name}")
    return solution_fn(**kwargs)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _clone_value(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return value
