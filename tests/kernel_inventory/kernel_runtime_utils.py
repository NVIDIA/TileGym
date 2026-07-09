# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import importlib.util
import inspect
import itertools
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

# Correctness coverage should select a stable config before importing cuda.tile.
os.environ.setdefault("DISABLE_TUNE", "1")
os.environ.setdefault("TILEGYM_DISABLE_AUTOTUNE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tilegym-triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import inventory modules without running tilegym/__init__.py. Runtime tests
# import the individual solution modules directly from Solution.spec.entry_point.
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory import iter_kernel_definition_paths
    from tilegym.kernel_inventory import iter_solution_paths_for_definition
    from tilegym.kernel_inventory import load_json
    from tilegym.kernel_inventory import validate_definition
    from tilegym.kernel_inventory import validate_solution
    from tilegym.kernel_inventory import validate_solution_entry_point
    from tilegym.kernel_inventory.return_contract import CAPTURE_RETURN_NAME
    from tilegym.kernel_inventory.return_contract import instrument_reference_returns
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)

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


class _CapturedReferenceReturn:
    """Reference value paired with the executed AST-derived output-name tree."""

    def __init__(self, value: Any, contract: Any):
        self.value = value
        self.contract = contract


def all_definition_solution_cases() -> list[Any]:
    """Return runtime parameter pairs for every discoverable inventory Definition."""
    return definition_solution_cases_for_submodule(None)


def definition_solution_cases_for_submodule(submodule: str | Path | None) -> list[Any]:
    """Return runtime parameter pairs for one inventory submodule or the full catalog."""
    search_root = _kernel_submodule_root(submodule)
    cases = []
    definition_paths = (
        iter_kernel_definition_paths(REPO_ROOT)
        if search_root == REPO_ROOT
        else sorted(search_root.glob("kernel_definitions/*.json"))
    )
    for definition_path in definition_paths:
        solution_paths = list(iter_solution_paths_for_definition(definition_path))
        if not solution_paths:
            raise ValueError(f"Definition has no checked-in Solutions: {definition_path}")
        for solution_path in solution_paths:
            solution = load_json(solution_path)
            cases.append(pytest.param(definition_path, solution_path, id=solution["name"]))
    if search_root != REPO_ROOT and not cases:
        raise ValueError(f"Kernel submodule has no Definition/Solution pairs: {search_root}")
    return cases


def _kernel_submodule_root(submodule: str | Path | None) -> Path:
    """Resolve an inventory-bearing submodule below ``src/tilegym``."""
    if submodule is None or str(submodule) == "":
        return REPO_ROOT

    raw_path = Path(submodule)
    if raw_path.is_absolute():
        candidates = (raw_path,)
    else:
        candidates = (
            Path.cwd() / raw_path,
            REPO_ROOT / raw_path,
            REPO_ROOT / "src/tilegym" / raw_path,
            REPO_ROOT / "src/tilegym/transformers" / raw_path,
            REPO_ROOT / "src/tilegym/suites" / raw_path,
        )
    submodule_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    submodule_root = submodule_root.resolve()
    tilegym_root = (REPO_ROOT / "src/tilegym").resolve()
    try:
        submodule_root.relative_to(tilegym_root)
    except ValueError as exc:
        raise ValueError(f"Kernel submodule must live under {tilegym_root}: {submodule}") from exc

    if not submodule_root.is_dir():
        raise ValueError(f"Kernel submodule does not exist: {submodule}")
    missing_directories = [
        directory
        for directory in ("kernel_definitions", "kernel_solutions")
        if not (submodule_root / directory).is_dir()
    ]
    if missing_directories:
        raise ValueError(
            f"Kernel submodule must contain kernel_definitions and kernel_solutions: {submodule_root} "
            f"(missing {', '.join(missing_directories)})"
        )
    return submodule_root


@contextmanager
def _isolated_solution_modules():
    """Temporarily isolate TileGym and FlashInfer imports used by Solution modules."""
    prefixes = ("tilegym", "flashinfer")
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg

    # Importing tilegym.backend probes the optional FlashInfer backend. The
    # inventory harness does not validate it, so avoid its JIT initialization.
    flashinfer_pkg = types.ModuleType("flashinfer")
    flashinfer_pkg.single_prefill_with_kv_cache = object()
    sys.modules["flashinfer"] = flashinfer_pkg
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def run_definition_solution_runtime(definition_path: Path, solution_path: Path) -> None:
    """Validate one Solution against its Definition reference or error contract."""
    with _isolated_solution_modules():
        _run_definition_solution_runtime(definition_path, solution_path)


def _run_definition_solution_runtime(definition_path: Path, solution_path: Path) -> None:
    """Run one Definition/Solution check with package imports isolated."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("flashinfer_bench")
    pytest.importorskip("cuda.tile")

    definition = load_json(definition_path)
    solution = load_json(solution_path)
    validate_definition(definition)
    validate_solution(solution, repo_root=REPO_ROOT)
    validate_solution_entry_point(solution, repo_root=REPO_ROOT)
    reference_fn = _load_reference(definition, definition_path)
    solution_fn = _load_solution_entry(solution)
    _assert_matching_entry_signatures(definition, reference_fn, solution, solution_fn)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for kernel inventory runtime checks")
    _skip_if_solution_does_not_target_current_compute_capability(solution, torch)

    device = torch.device("cuda")
    torch.manual_seed(2026)
    axes = _axis_values(definition)
    base_inputs = _make_inputs(definition, torch, device, axes)
    for boolean_assignment in _boolean_branch_assignments(definition, axes, base_inputs):
        inputs = dict(base_inputs)
        inputs.update(boolean_assignment)
        _run_runtime_branch(definition, reference_fn, solution_fn, inputs, axes, torch)


def _run_runtime_branch(
    definition: dict[str, Any],
    reference_fn: Any,
    solution_fn: Any,
    inputs: dict[str, Any],
    axes: dict[str, int],
    torch: Any,
) -> None:
    """Compare reference and Solution for one concrete Boolean branch."""
    reference_inputs = {name: _clone_value(value) for name, value in inputs.items()}
    solution_inputs = {name: _clone_value(value) for name, value in inputs.items()}
    branch = _boolean_branch_label(inputs)

    if "runtime:unsupported" in definition.get("tags", []):
        _assert_unsupported_solution_matches_reference(
            definition,
            reference_fn,
            reference_inputs,
            solution_fn,
            solution_inputs,
        )
        return

    reference_result = _call_entry_strictly(reference_fn, reference_inputs, "Definition.reference")
    solution_result = _call_entry_strictly(solution_fn, solution_inputs, "Solution entry point")
    torch.cuda.synchronize()

    assert isinstance(reference_result, _CapturedReferenceReturn), (
        f"{definition['name']} {branch}: Definition.reference return was not instrumented"
    )
    _assert_return_contract(
        solution_result,
        reference_result.value,
        reference_result.contract,
        definition["outputs"],
        axes,
        torch,
        f"{definition['name']} {branch}",
    )

    for name in MUTATED_INPUTS_BY_DEFINITION.get(definition["name"], ()):
        torch.testing.assert_close(
            solution_inputs[name],
            reference_inputs[name],
            rtol=2e-2,
            atol=2e-2,
            msg=lambda msg: f"{definition['name']} {branch} mutated input {name} mismatch\n{msg}",
        )


def _assert_unsupported_solution_matches_reference(
    definition: dict[str, Any],
    reference_fn: Any,
    reference_inputs: dict[str, Any],
    solution_fn: Any,
    solution_inputs: dict[str, Any],
) -> None:
    """Verify an intentionally unsupported Solution raises the reference error type."""
    reference_error = _capture_runtime_error(
        lambda: _call_entry_strictly(reference_fn, reference_inputs, "Definition.reference"),
        f"{definition['name']} reference",
    )
    solution_error = _capture_runtime_error(
        lambda: _call_entry_strictly(solution_fn, solution_inputs, "Solution entry point"),
        f"{definition['name']} Solution",
    )
    assert type(solution_error) is type(reference_error), (
        f"{definition['name']}: deprecated Solution raised {type(solution_error).__name__}; "
        f"reference raised {type(reference_error).__name__}"
    )


def _capture_runtime_error(call: Any, label: str) -> Exception:
    """Run ``call`` and return its required exception."""
    try:
        call()
    except Exception as exc:
        return exc
    pytest.fail(f"{label} must raise because the Definition is tagged runtime:unsupported")


def _axis_values(definition: dict[str, Any]) -> dict[str, int]:
    """Build deterministic concrete axis values for a Definition runtime case."""
    values = {name: axis["value"] for name, axis in definition["axes"].items() if axis.get("type") == "const"}
    for name, axis in definition["axes"].items():
        if axis.get("type") == "var":
            defaults = DEFAULT_AXIS_VALUES
            values[name] = defaults.get(name, 4)
    if "T_padded" in values and "T" in values and "K" in values:
        values["T_padded"] = values["T"] + values["K"] - 1
    return values


def _make_inputs(definition: dict[str, Any], torch: Any, device: Any, axes: dict[str, int]) -> dict[str, Any]:
    """Create seeded representative inputs that conform to a Definition."""
    inputs = {}
    for name, spec in definition["inputs"].items():
        inputs[name] = _make_input(name, spec, axes, torch, device)
    _satisfy_boolean_constraints(definition, inputs, axes)
    return inputs


def _boolean_branch_assignments(
    definition: dict[str, Any],
    axes: dict[str, int] | None = None,
    inputs: dict[str, Any] | None = None,
) -> list[dict[str, bool]]:
    """Enumerate scalar Boolean assignments satisfying applicable constraints."""
    boolean_inputs = [
        name for name, spec in definition["inputs"].items() if spec["shape"] is None and spec["dtype"] == "bool"
    ]
    if not boolean_inputs:
        return [{}]
    constraints = [
        constraint
        for constraint in definition.get("constraints", ())
        if set(boolean_inputs) & _constraint_names(constraint)
    ]
    assignments = []
    for values in itertools.product((False, True), repeat=len(boolean_inputs)):
        assignment = dict(zip(boolean_inputs, values, strict=True))
        candidate = dict(inputs or {})
        candidate.update(assignment)
        if _constraints_hold(constraints, candidate, axes or {}):
            assignments.append(assignment)
    if not assignments:
        pytest.fail(f"{definition['name']}: no Boolean input assignment satisfies Definition.constraints")
    return assignments


def _boolean_branch_label(inputs: dict[str, Any]) -> str:
    """Describe the concrete scalar Boolean branch in assertion messages."""
    values = [f"{name}={value}" for name, value in inputs.items() if isinstance(value, bool)]
    return f"[{', '.join(values)}]" if values else "[no Boolean inputs]"


def _satisfy_boolean_constraints(definition: dict[str, Any], inputs: dict[str, Any], axes: dict[str, int]) -> None:
    """Choose scalar Boolean inputs that satisfy evaluable Definition constraints."""
    inputs.update(_boolean_branch_assignments(definition, axes, inputs)[0])


def _constraint_names(constraint: str) -> set[str]:
    """Return variable names in one valid Python constraint expression."""
    try:
        tree = ast.parse(constraint, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _constraints_hold(constraints: Any, inputs: dict[str, Any], axes: dict[str, int]) -> bool:
    """Evaluate constraints that depend only on axes and Python scalar inputs."""
    context = dict(axes)
    context.update({name: value for name, value in inputs.items() if isinstance(value, bool | float | int)})
    for constraint in constraints:
        try:
            result = eval(compile(constraint, "<Definition.constraints>", "eval"), {"__builtins__": {}}, context)
        except (NameError, SyntaxError):
            continue
        if not result:
            return False
    return True


def _make_input(name: str, spec: dict[str, Any], axes: dict[str, int], torch: Any, device: Any) -> Any:
    """Create one scalar or tensor input from its schema specification."""
    dtype = spec["dtype"]
    shape_spec = spec["shape"]
    if shape_spec is None:
        if name == "scale":
            return axes.get("K", axes.get("D", 64)) ** -0.5
        if dtype == "bool":
            return False
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
    """Resolve an inventory dtype string to its torch dtype."""
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


def _assert_output_matches_spec(value: Any, spec: dict[str, Any], axes: dict[str, int], torch: Any, label: str) -> None:
    """Check one concrete tensor output against its Definition TensorSpec."""
    expected_shape = () if spec["shape"] is None else tuple(axes[axis] for axis in spec["shape"])
    assert tuple(value.shape) == expected_shape, (
        f"{label} has shape {tuple(value.shape)}; Definition declares {expected_shape}"
    )
    expected_dtype = _torch_dtype(spec["dtype"], torch)
    assert value.dtype == expected_dtype, f"{label} has dtype {value.dtype}; Definition declares {expected_dtype}"


def _assert_return_contract(
    actual: Any,
    expected: Any,
    contract: Any,
    output_specs: dict[str, dict[str, Any]],
    axes: dict[str, int],
    torch: Any,
    label: str,
) -> None:
    """Recursively compare values using the executed reference return-name tree."""
    if contract is None:
        assert actual is expected is None, f"{label}: expected matching None return values"
        return
    if isinstance(contract, tuple):
        assert isinstance(actual, tuple) and isinstance(expected, tuple), (
            f"{label}: reference and Solution must both return a tuple"
        )
        assert len(actual) == len(expected) == len(contract), (
            f"{label}: nested return arity does not match reference contract"
        )
        for index, (actual_item, expected_item, item_contract) in enumerate(
            zip(actual, expected, contract, strict=True)
        ):
            _assert_return_contract(
                actual_item,
                expected_item,
                item_contract,
                output_specs,
                axes,
                torch,
                f"{label}[{index}]",
            )
        return

    assert isinstance(contract, str) and contract in output_specs, f"{label}: unknown output contract {contract!r}"
    if actual is None or expected is None:
        assert actual is expected is None, f"{label} output {contract}: tensor/None mismatch"
        return
    spec = output_specs[contract]
    _assert_output_matches_spec(actual, spec, axes, torch, f"Solution entry point output {contract}")
    _assert_output_matches_spec(expected, spec, axes, torch, f"Definition.reference output {contract}")
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
        msg=lambda msg: f"{label} output {contract} mismatch\n{msg}",
    )


def _skip_if_solution_does_not_target_current_compute_capability(solution: dict[str, Any], torch: Any) -> None:
    """Skip only when a Solution explicitly excludes the active GPU capability."""
    target_hardware = solution["spec"].get("target_hardware", [])
    current_hardware = _current_compute_capability_label(torch)
    if target_hardware and current_hardware not in target_hardware:
        pytest.skip(
            f"Solution targets {target_hardware}; current compute capability is {current_hardware} "
            f"({torch.cuda.get_device_name(0)})"
        )


def _current_compute_capability_label(torch: Any) -> str:
    """Return the active CUDA compute capability as an ``SM<major><minor>`` label."""
    major, minor = torch.cuda.get_device_capability(0)
    return f"SM{major}{minor}"


def _load_reference(definition: dict[str, Any], definition_path: Path) -> Any:
    """Compile an AST-instrumented Definition reference ``run`` callable."""
    module = instrument_reference_returns(
        definition["reference"],
        list(definition["outputs"]),
        allow_no_return="runtime:unsupported" in definition.get("tags", []),
    )
    namespace: dict[str, Any] = {
        CAPTURE_RETURN_NAME: lambda value, contract: _CapturedReferenceReturn(value, contract),
    }
    exec(compile(module, str(definition_path), "exec"), namespace)
    return namespace["run"]


def _load_solution_entry(solution: dict[str, Any]) -> Any:
    """Load the callable named by a Solution entry point without package side effects."""
    file_path, symbol = solution["spec"]["entry_point"].split("::", 1)
    source_path = REPO_ROOT / file_path
    module_name = _solution_module_name(source_path)
    _ensure_solution_parent_packages(source_path, module_name)
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return getattr(module, symbol)


def _solution_module_name(source_path: Path) -> str:
    """Return the package-qualified name for a checked-in TileGym module."""
    relative_path = source_path.relative_to(REPO_ROOT / "src").with_suffix("")
    return ".".join(relative_path.parts)


def _ensure_solution_parent_packages(source_path: Path, module_name: str) -> None:
    """Install lightweight package parents so Solution modules can use relative imports.

    Importing a backend package can execute its ``__init__.py`` and import every
    implementation. Runtime validation deliberately loads one entry point at a
    time, so it constructs only the package parents needed for Python's
    relative-import resolver.
    """
    source_root = REPO_ROOT / "src"
    relative_path = source_path.relative_to(source_root)
    package_parts = module_name.split(".")[:-1]
    for depth in range(1, len(package_parts) + 1):
        package_name = ".".join(package_parts[:depth])
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_root.joinpath(*relative_path.parts[:depth]))]
        package.__package__ = package_name
        sys.modules[package_name] = package


def _assert_matching_entry_signatures(
    definition: dict[str, Any], reference_fn: Any, solution: dict[str, Any], solution_fn: Any
) -> None:
    """Require Definition reference and Solution entry points to expose one call contract."""
    reference_signature = _call_signature(reference_fn)
    solution_signature = _call_signature(solution_fn)
    assert reference_signature == solution_signature, (
        f"{definition['name']}: Definition.reference run signature {reference_signature} does not match "
        f"Solution entry point {solution['spec']['entry_point']} signature {solution_signature}"
    )


def _call_signature(callable_: Any) -> inspect.Signature:
    """Return a callable's public invocation signature without type-only annotations."""
    signature = inspect.signature(callable_)
    return signature.replace(
        parameters=[
            parameter.replace(annotation=inspect.Parameter.empty) for parameter in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )


def _call_entry_strictly(entry_fn: Any, inputs: dict[str, Any], label: str) -> Any:
    """Call an entry with required parameters positional and defaulted parameters by keyword."""
    signature = _call_signature(entry_fn)
    positional_args = []
    keyword_args = {}
    accepted_names = set()
    accepts_arbitrary_keywords = False

    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_arbitrary_keywords = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue

        accepted_names.add(name)
        if name not in inputs:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(f"{label} requires Definition input '{name}'")
            continue

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if parameter.default is not inspect.Parameter.empty:
                raise TypeError(f"{label} has an optional positional-only parameter '{name}', which is unsupported")
            positional_args.append(inputs[name])
        elif parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if parameter.default is inspect.Parameter.empty:
                positional_args.append(inputs[name])
            else:
                keyword_args[name] = inputs[name]
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_args[name] = inputs[name]

    unexpected_names = sorted(set(inputs) - accepted_names)
    if accepts_arbitrary_keywords:
        keyword_args.update({name: inputs[name] for name in unexpected_names})
        unexpected_names = []
    if unexpected_names:
        raise TypeError(f"{label} does not accept Definition inputs: {unexpected_names}")
    return entry_fn(*positional_args, **keyword_args)


def _as_outputs(value: Any) -> tuple[Any, ...]:
    """Normalize a return value to a tuple without changing its declared arity."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _clone_value(value: Any) -> Any:
    """Clone tensors while preserving scalar values for independent invocations."""
    if hasattr(value, "clone"):
        return value.clone()
    return value
