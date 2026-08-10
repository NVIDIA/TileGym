# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Backend-neutral launch contracts for raw inventory kernel entry points.

Definitions remain the single source of truth for tensor shapes and dtypes.
This module only describes how a raw kernel parameter obtains its runtime
value and how its launch grid is computed.  Backend-specific invocation (for
example Triton's ``kernel[grid]`` syntax) deliberately lives elsewhere.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import Literal
from typing import Protocol

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

BindingKind = Literal["input", "output", "axis", "tensor_shape", "tensor_stride", "tensor_numel", "literal"]
OutputInitialization = Literal["empty", "zeros", "ones"]
TensorView = Literal["flatten"]
GridSpec = tuple[int, ...] | str


class LaunchContractError(ValueError):
    """Raised when launch metadata is invalid or cannot be materialized."""


class LaunchArgumentBinding(BaseModel):
    """Bind one raw entry-point parameter to Definition-owned runtime data."""

    model_config = ConfigDict(extra="forbid")

    parameter: str = Field(pattern=r"^[A-Za-z_]\w*$")
    kind: BindingKind
    name: str | None = None
    dimension: int | None = None
    value: Any = None
    initialize: OutputInitialization | None = None
    initialize_from: str | None = None
    view: TensorView | None = None
    target_triton_backends: list[Literal["nvt", "oait"]] | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _require_literal_value_key(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and data.get("kind") == "literal" and "value" not in data:
            raise ValueError("A literal binding must contain a value (which may be null)")
        return data

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "LaunchArgumentBinding":
        tensor_derived = {"tensor_shape", "tensor_stride"}
        named = {"input", "output", "axis", "tensor_shape", "tensor_stride", "tensor_numel"}
        if self.kind in named and not self.name:
            raise ValueError(f"A {self.kind} binding requires name")
        if self.kind not in named and self.name is not None:
            raise ValueError(f"A {self.kind} binding cannot specify name")
        if self.kind in tensor_derived and self.dimension is None:
            raise ValueError(f"A {self.kind} binding requires dimension")
        if self.kind not in tensor_derived and self.dimension is not None:
            raise ValueError(f"A {self.kind} binding cannot specify dimension")
        if self.kind != "literal" and self.value is not None:
            raise ValueError(f"A {self.kind} binding cannot contain a literal value")
        if self.kind != "output" and self.initialize is not None:
            raise ValueError(f"A {self.kind} binding cannot specify output initialization")
        if self.kind != "output" and self.initialize_from is not None:
            raise ValueError(f"A {self.kind} binding cannot specify an output seed")
        if self.kind == "output" and self.initialize is not None and self.initialize_from is not None:
            raise ValueError("An output binding cannot specify both initialize and initialize_from")
        if self.initialize_from is not None and not self.initialize_from:
            raise ValueError("Output initialize_from must name a Definition input")
        if self.view is not None and self.kind not in {"input", "output"}:
            raise ValueError(f"A {self.kind} binding cannot specify a tensor view")
        if self.target_triton_backends is not None:
            if len(set(self.target_triton_backends)) != len(self.target_triton_backends):
                raise ValueError("Launch binding target_triton_backends entries must be unique")
            if set(self.target_triton_backends) == {"nvt", "oait"}:
                raise ValueError(
                    "Launch binding target_triton_backends must select a proper subset; "
                    "omit the field for an unconditional binding"
                )
        if self.kind == "literal":
            _validate_json_literal(self.value)
        return self


class RawKernelLaunch(BaseModel):
    """Serializable launch metadata attached to a raw-kernel Solution."""

    model_config = ConfigDict(extra="forbid")

    arguments: list[LaunchArgumentBinding] = Field(default_factory=list)
    grid: GridSpec

    @model_validator(mode="before")
    @classmethod
    def _normalize_grid(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        normalized = dict(data)
        grid = normalized.get("grid")
        if isinstance(grid, (list, tuple)):
            _validate_grid(tuple(grid))
            normalized["grid"] = tuple(grid)
        return normalized

    @model_validator(mode="after")
    def _validate_launch(self) -> "RawKernelLaunch":
        parameters = [binding.parameter for binding in self.arguments]
        duplicates = sorted({name for name in parameters if parameters.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate launch parameter bindings: {duplicates}")
        if isinstance(self.grid, tuple):
            _validate_grid(self.grid)
        elif self.grid.count("::") != 1:
            raise ValueError('Callable grid must use the repo-local "path.py::symbol" form')
        return self


@dataclass(frozen=True)
class RawParameter:
    """One parameter recovered from a raw entry point's Python AST."""

    name: str
    kind: Literal["positional_only", "positional_or_keyword", "keyword_only"]
    required: bool
    provided_by_autotune: bool = False
    possibly_provided_by_autotune: bool = False
    provided_by_heuristic: bool = False


@dataclass(frozen=True)
class RawEntrySignature:
    """Static signature of a top-level raw kernel entry point."""

    entry_point: str
    parameters: tuple[RawParameter, ...]


@dataclass(frozen=True)
class LaunchContext:
    """Definition-shaped runtime data available to launch bindings and grids."""

    definition_name: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    axes: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "axes", MappingProxyType(dict(self.axes)))

    def tensor(self, name: str) -> Any:
        if name in self.inputs:
            return self.inputs[name]
        if name in self.outputs:
            return self.outputs[name]
        raise LaunchContractError(f"No runtime tensor named '{name}'")

    def shape(self, name: str, dimension: int) -> int:
        shape = _runtime_shape(self.tensor(name), name)
        try:
            return int(shape[dimension])
        except IndexError as exc:
            raise LaunchContractError(f"Tensor '{name}' has no shape dimension {dimension}") from exc

    def stride(self, name: str, dimension: int) -> int:
        tensor = self.tensor(name)
        stride = getattr(tensor, "stride", None)
        if not callable(stride):
            raise LaunchContractError(f"Runtime tensor '{name}' does not expose stride()")
        try:
            return int(stride(dimension))
        except (IndexError, RuntimeError) as exc:
            raise LaunchContractError(f"Tensor '{name}' has no stride dimension {dimension}") from exc

    def numel(self, name: str) -> int:
        tensor = self.tensor(name)
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            return int(numel())
        result = 1
        for extent in _runtime_shape(tensor, name):
            result *= int(extent)
        return result


class OutputAllocator(Protocol):
    """Framework adapter used to allocate an output from a Definition spec."""

    def __call__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        spec: Mapping[str, Any],
    ) -> Any: ...


@dataclass(frozen=True)
class MaterializedLaunch:
    """Concrete, backend-neutral values ready for an invocation adapter."""

    grid: tuple[int, ...]
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


def make_launch_context(
    definition: Any,
    inputs: Mapping[str, Any],
    *,
    outputs: Mapping[str, Any] | None = None,
    axes: Mapping[str, int] | None = None,
) -> LaunchContext:
    """Build a launch context and derive Definition axes from runtime inputs."""
    data = _definition_data(definition)
    definition_inputs = _mapping_field(data, "inputs")
    unknown_inputs = sorted(set(inputs) - set(definition_inputs))
    missing_inputs = sorted(set(definition_inputs) - set(inputs))
    if unknown_inputs:
        raise LaunchContractError(f"Runtime inputs are not present in the Definition: {unknown_inputs}")
    if missing_inputs:
        raise LaunchContractError(f"Definition inputs have no runtime value: {missing_inputs}")

    resolved_axes = _definition_constant_axes(data)
    for name, spec in definition_inputs.items():
        shape_spec = _field(spec, "shape")
        if shape_spec is None:
            continue
        runtime_shape = _runtime_shape(inputs[name], name)
        if len(runtime_shape) != len(shape_spec):
            raise LaunchContractError(
                f"Runtime input '{name}' rank {len(runtime_shape)} does not match Definition rank {len(shape_spec)}"
            )
        for axis_name, extent in zip(shape_spec, runtime_shape):
            if not isinstance(axis_name, str):
                raise LaunchContractError(f"Definition shape for '{name}' contains non-axis value {axis_name!r}")
            _merge_axis(resolved_axes, axis_name, int(extent), f"runtime input '{name}'")

    for axis_name, extent in (axes or {}).items():
        _merge_axis(resolved_axes, axis_name, int(extent), "explicit launch axes")

    declared_axes = _mapping_field(data, "axes")
    unknown_axes = sorted(set(resolved_axes) - set(declared_axes))
    if unknown_axes:
        raise LaunchContractError(f"Resolved axes are not declared by the Definition: {unknown_axes}")

    return LaunchContext(
        definition_name=str(_field(data, "name")),
        inputs=inputs,
        outputs=outputs or {},
        axes=resolved_axes,
    )


def allocate_definition_outputs(
    definition: Any,
    axes: Mapping[str, int],
    allocator: OutputAllocator,
    *,
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Allocate outputs using only Definition-owned shape and dtype metadata."""
    data = _definition_data(definition)
    output_specs = _mapping_field(data, "outputs")
    requested = list(output_specs) if names is None else list(names)
    unknown = sorted(set(requested) - set(output_specs))
    if unknown:
        raise LaunchContractError(f"Requested outputs are not present in the Definition: {unknown}")

    allocated: dict[str, Any] = {}
    for name in requested:
        spec = _as_mapping(output_specs[name], f"Definition output '{name}'")
        shape_spec = _field(spec, "shape")
        shape = () if shape_spec is None else tuple(_axis_extent(axis, axes, name) for axis in shape_spec)
        dtype = _field(spec, "dtype")
        if not isinstance(dtype, str) or not dtype:
            raise LaunchContractError(f"Definition output '{name}' has no valid dtype")
        allocated[name] = allocator(name=name, shape=shape, dtype=dtype, spec=spec)
    return allocated


def validate_launch_contract(
    launch: RawKernelLaunch | Mapping[str, Any],
    definition: Any,
    entry_point: str,
    repo_root: str | Path,
) -> RawEntrySignature:
    """Validate bindings against a Definition and the raw entry point AST."""
    model = launch if isinstance(launch, RawKernelLaunch) else RawKernelLaunch.model_validate(launch)
    data = _definition_data(definition)
    definition_inputs = _mapping_field(data, "inputs")
    definition_outputs = _mapping_field(data, "outputs")
    definition_axes = _mapping_field(data, "axes")
    tensors = set(definition_inputs) | set(definition_outputs)

    for binding in model.arguments:
        if binding.kind == "input" and binding.name not in definition_inputs:
            raise LaunchContractError(
                f"Binding '{binding.parameter}' references unknown Definition input '{binding.name}'"
            )
        if binding.kind == "output" and binding.name not in definition_outputs:
            raise LaunchContractError(
                f"Binding '{binding.parameter}' references unknown Definition output '{binding.name}'"
            )
        if binding.view is not None:
            assert binding.name is not None
            tensor_specs = definition_inputs if binding.kind == "input" else definition_outputs
            tensor_spec = tensor_specs[binding.name]
            if not _field(tensor_spec, "shape"):
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' cannot apply a tensor view to scalar Definition tensor "
                    f"'{binding.name}'"
                )
        if binding.kind == "output" and binding.initialize_from is not None:
            seed_name = binding.initialize_from
            if seed_name not in definition_inputs:
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' seeds its output from unknown Definition input '{seed_name}'"
                )
            output_spec = definition_outputs[binding.name]
            seed_spec = definition_inputs[seed_name]
            if _field(output_spec, "shape") != _field(seed_spec, "shape"):
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' output '{binding.name}' and seed input '{seed_name}' "
                    "must have identical shapes"
                )
            if _field(output_spec, "dtype") != _field(seed_spec, "dtype"):
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' output '{binding.name}' and seed input '{seed_name}' "
                    "must have identical dtypes"
                )
        if binding.kind == "axis" and binding.name not in definition_axes:
            raise LaunchContractError(
                f"Binding '{binding.parameter}' references unknown Definition axis '{binding.name}'"
            )
        if binding.kind.startswith("tensor_") and binding.name not in tensors:
            raise LaunchContractError(
                f"Binding '{binding.parameter}' references unknown Definition tensor '{binding.name}'"
            )
        if binding.kind in {"tensor_shape", "tensor_stride"}:
            spec = definition_inputs.get(binding.name, definition_outputs.get(binding.name))
            shape = _field(spec, "shape")
            if shape is None:
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' cannot derive a dimension from scalar tensor '{binding.name}'"
                )
            dimension = binding.dimension
            assert dimension is not None
            if not -len(shape) <= dimension < len(shape):
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' dimension {dimension} is outside tensor '{binding.name}' rank {len(shape)}"
                )

    output_bindings = [binding.name for binding in model.arguments if binding.kind == "output"]
    duplicate_outputs = sorted({name for name in output_bindings if output_bindings.count(name) > 1})
    if duplicate_outputs:
        raise LaunchContractError(f"Definition outputs have duplicate raw destination bindings: {duplicate_outputs}")
    bound_outputs = set(output_bindings)
    missing_outputs = sorted(set(definition_outputs) - bound_outputs)
    if missing_outputs:
        raise LaunchContractError(f"Definition outputs have no raw destination binding: {missing_outputs}")

    signature = inspect_raw_entry_signature(entry_point, repo_root)
    parameter_names = {parameter.name for parameter in signature.parameters}
    binding_names = {binding.parameter for binding in model.arguments}
    unknown_parameters = sorted(binding_names - parameter_names)
    autotune_conflicts = sorted(
        parameter.name
        for parameter in signature.parameters
        if parameter.provided_by_autotune and parameter.name in binding_names
    )
    conditional_autotune_parameters = {
        parameter.name
        for parameter in signature.parameters
        if parameter.possibly_provided_by_autotune and not parameter.provided_by_autotune
    }
    scoped_bindings = {binding.parameter for binding in model.arguments if binding.target_triton_backends is not None}
    invalid_scoped_bindings = sorted(scoped_bindings - conditional_autotune_parameters)
    heuristic_parameters = {parameter.name for parameter in signature.parameters if parameter.provided_by_heuristic}
    bound_heuristics = heuristic_parameters & binding_names
    missing_parameters = [
        parameter.name
        for parameter in signature.parameters
        if parameter.required and parameter.name not in binding_names
    ]
    if unknown_parameters:
        raise LaunchContractError(f"Launch bindings reference unknown raw entry parameters: {unknown_parameters}")
    if autotune_conflicts:
        raise LaunchContractError(
            f"Launch bindings must omit parameters supplied by autotune configs: {autotune_conflicts}"
        )
    if invalid_scoped_bindings:
        raise LaunchContractError(
            "Compiler-scoped launch bindings are valid only for parameters supplied by some but not all "
            f"reachable Triton autotune configs: {invalid_scoped_bindings}"
        )
    if bound_heuristics and bound_heuristics != heuristic_parameters:
        missing_heuristics = sorted(heuristic_parameters - bound_heuristics)
        raise LaunchContractError(
            "Launch bindings that override Triton heuristics must bind every heuristic-owned parameter; "
            f"missing {missing_heuristics}"
        )
    if missing_parameters:
        raise LaunchContractError(f"Required raw entry parameters have no launch binding: {missing_parameters}")

    if isinstance(model.grid, str):
        inspect_grid_callable(model.grid, repo_root)
    return signature


def inspect_raw_entry_signature(entry_point: str, repo_root: str | Path) -> RawEntrySignature:
    """Read a top-level Python entry point signature without importing it."""
    path, symbol, module = _entry_ast(entry_point, repo_root)
    function = _find_top_level_function(module, symbol, path)
    if function.args.vararg is not None or function.args.kwarg is not None:
        raise LaunchContractError(f"Raw entry point '{entry_point}' cannot use *args or **kwargs")

    positional = [*function.args.posonlyargs, *function.args.args]
    required_positional_count = len(positional) - len(function.args.defaults)
    autotune_parameters, possible_autotune_parameters = _autotune_config_parameter_ownership(function, module)
    heuristic_parameters = _triton_heuristic_parameters(function)
    decorator_parameters = autotune_parameters | heuristic_parameters
    parameters = [
        RawParameter(
            argument.arg,
            "positional_only" if index < len(function.args.posonlyargs) else "positional_or_keyword",
            index < required_positional_count and argument.arg not in decorator_parameters,
            argument.arg in autotune_parameters,
            argument.arg in possible_autotune_parameters,
            argument.arg in heuristic_parameters,
        )
        for index, argument in enumerate(positional)
    ]
    parameters.extend(
        RawParameter(
            argument.arg,
            "keyword_only",
            default is None and argument.arg not in decorator_parameters,
            argument.arg in autotune_parameters,
            argument.arg in possible_autotune_parameters,
            argument.arg in heuristic_parameters,
        )
        for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
    )
    return RawEntrySignature(entry_point=entry_point, parameters=tuple(parameters))


def _triton_heuristic_parameters(function: ast.FunctionDef) -> set[str]:
    """Return parameters supplied by outer ``@triton.heuristics`` decorators."""
    keys: set[str] = set()
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or _ast_name(decorator.func).rsplit(".", 1)[-1] != "heuristics":
            continue
        values = (
            decorator.args[0]
            if decorator.args
            else next(
                (keyword.value for keyword in decorator.keywords if keyword.arg == "values"),
                None,
            )
        )
        if not isinstance(values, ast.Dict):
            continue
        keys.update(key.value for key in values.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
    parameter_names = {
        argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    }
    return keys & parameter_names


def _autotune_config_parameter_ownership(
    function: ast.FunctionDef,
    module: ast.Module,
) -> tuple[set[str], set[str]]:
    """Return kernel parameters owned by all and by any reachable Triton configs."""
    expressions = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or _ast_name(decorator.func).rsplit(".", 1)[-1] != "autotune":
            continue
        expressions.extend(keyword.value for keyword in decorator.keywords if keyword.arg == "configs")
    if not expressions:
        return set(), set()

    definitions: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            definitions[node.target.id] = node.value

    def local_definitions(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[ast.AST]]:
        local: dict[str, list[ast.AST]] = {}
        for child in ast.walk(function):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        local.setdefault(target.id, []).append(child.value)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.value is not None:
                local.setdefault(child.target.id, []).append(child.value)
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.attr in {"clear", "pop", "popitem", "__delitem__"}
            ):
                local.setdefault(child.func.value.id, []).append(ast.Constant(None))
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.attr in {"append", "extend", "insert"}
            ):
                argument_index = 1 if child.func.attr == "insert" else 0
                if len(child.args) > argument_index:
                    value = child.args[argument_index]
                    local.setdefault(child.func.value.id, []).append(
                        value if child.func.attr == "extend" else ast.List(elts=[value], ctx=ast.Load())
                    )
                else:
                    local.setdefault(child.func.value.id, []).append(ast.Constant(None))
            elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                local.setdefault(child.target.id, []).append(child.value)
            elif isinstance(child, ast.Delete):
                for target in child.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        local.setdefault(target.value.id, []).append(ast.Constant(None))

        def sequence_elements(node: ast.AST) -> list[ast.AST]:
            if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
                elements: list[ast.AST] = []
                for element in node.elts:
                    if isinstance(element, ast.Starred):
                        elements.extend(sequence_elements(element.value))
                    else:
                        elements.append(element)
                return elements
            if isinstance(node, ast.Name):
                targets = local.get(node.id)
                if targets is None:
                    target = definitions.get(node.id)
                    targets = [target] if target is not None and not isinstance(target, ast.FunctionDef) else []
                return [element for target in targets for element in sequence_elements(target)] or [ast.Constant(None)]
            if isinstance(node, ast.IfExp):
                return [*sequence_elements(node.body), *sequence_elements(node.orelse)]
            return [ast.Constant(None)]

        for child in ast.walk(function):
            if not isinstance(child, ast.comprehension):
                continue
            elements = sequence_elements(child.iter)
            if isinstance(child.target, ast.Name):
                local.setdefault(child.target.id, []).extend(elements)
            elif isinstance(child.target, (ast.List, ast.Tuple)):
                for element in elements:
                    if not isinstance(element, (ast.List, ast.Tuple)) or len(element.elts) != len(child.target.elts):
                        for target in child.target.elts:
                            if isinstance(target, ast.Name):
                                local.setdefault(target.id, []).append(ast.Constant(None))
                        continue
                    for target, value in zip(child.target.elts, element.elts):
                        if isinstance(target, ast.Name):
                            local.setdefault(target.id, []).append(value)
        return local

    def mapping_key_sets(
        node: ast.AST,
        local: dict[str, list[ast.AST]],
        resolving: frozenset[str] = frozenset(),
    ) -> list[set[str]]:
        """Return one statically safe key set for each possible mapping value."""
        if isinstance(node, ast.Dict):
            alternatives = [
                {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            ]
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    continue
                expanded = mapping_key_sets(value, local, resolving)
                alternatives = [known | extra for known in alternatives for extra in expanded]
            return alternatives
        if isinstance(node, ast.IfExp):
            return [*mapping_key_sets(node.body, local, resolving), *mapping_key_sets(node.orelse, local, resolving)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = mapping_key_sets(node.left, local, resolving)
            right = mapping_key_sets(node.right, local, resolving)
            return [left_keys | right_keys for left_keys in left for right_keys in right]
        if isinstance(node, ast.Name) and node.id not in resolving:
            targets = local.get(node.id)
            if targets is None:
                target = definitions.get(node.id)
                targets = [target] if target is not None and not isinstance(target, ast.FunctionDef) else []
            return [keys for target in targets for keys in mapping_key_sets(target, local, resolving | {node.id})] or [
                set()
            ]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in resolving:
            target = definitions.get(node.func.id)
            if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target_local = local_definitions(target)
                returns = [child.value for child in ast.walk(target) if isinstance(child, ast.Return) and child.value]
                return [
                    keys
                    for returned in returns
                    for keys in mapping_key_sets(returned, target_local, resolving | {node.func.id})
                ] or [set()]
        return [set()]

    def config_sequence_key_sets(
        node: ast.AST,
        local: dict[str, list[ast.AST]],
        resolving: frozenset[str] = frozenset(),
    ) -> list[set[str]]:
        """Return mapping keys for every possible config in a sequence expression."""
        if isinstance(node, ast.Call) and _ast_name(node.func).rsplit(".", 1)[-1] == "Config":
            mapping = node.args[0] if node.args else None
            return mapping_key_sets(mapping, local) if mapping is not None else [set()]
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            alternatives = [
                keys
                for element in node.elts
                for keys in config_sequence_key_sets(
                    element.value if isinstance(element, ast.Starred) else element,
                    local,
                    resolving,
                )
            ]
            return alternatives or [set()]
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
            return config_sequence_key_sets(node.elt, local, resolving)
        if isinstance(node, ast.IfExp):
            return [
                *config_sequence_key_sets(node.body, local, resolving),
                *config_sequence_key_sets(node.orelse, local, resolving),
            ]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return [
                *config_sequence_key_sets(node.left, local, resolving),
                *config_sequence_key_sets(node.right, local, resolving),
            ]
        if isinstance(node, ast.Name) and node.id not in resolving:
            targets = local.get(node.id)
            if targets is None:
                target = definitions.get(node.id)
                targets = [target] if target is not None and not isinstance(target, ast.FunctionDef) else []
            return [
                keys for target in targets for keys in config_sequence_key_sets(target, local, resolving | {node.id})
            ] or [set()]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in resolving:
            target = definitions.get(node.func.id)
            if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target_local = local_definitions(target)
                returns = [child.value for child in ast.walk(target) if isinstance(child, ast.Return) and child.value]
                return [
                    keys
                    for returned in returns
                    for keys in config_sequence_key_sets(returned, target_local, resolving | {node.func.id})
                ] or [set()]
        return [set()]

    config_key_sets = [keys for expression in expressions for keys in config_sequence_key_sets(expression, local={})]
    keys = set.intersection(*config_key_sets) if config_key_sets else set()
    possible_keys = set.union(*config_key_sets) if config_key_sets else set()
    parameter_names = {
        argument.arg for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
    }
    return keys & parameter_names, possible_keys & parameter_names


def inspect_grid_callable(entry_point: str, repo_root: str | Path) -> None:
    """Statically validate a repository-local ``grid(context, meta=None)``."""
    path, symbol, module = _entry_ast(entry_point, repo_root)
    function = _find_top_level_function(module, symbol, path)
    args = function.args
    if (
        args.posonlyargs
        or [argument.arg for argument in args.args] != ["context", "meta"]
        or len(args.defaults) != 1
        or not isinstance(args.defaults[0], ast.Constant)
        or args.defaults[0].value is not None
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
    ):
        raise LaunchContractError(
            f"Grid callable '{entry_point}' must have the exact signature grid(context, meta=None)"
        )
    if any(
        isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(function)
        if node is not function
    ):
        raise LaunchContractError(f"Grid callable '{entry_point}' cannot contain lambdas or nested functions")
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function)):
        raise LaunchContractError(f"Grid callable '{entry_point}' cannot import modules inside the function body")
    imports = _validated_grid_imports(module, path)
    bound_names = {argument.arg for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    bound_names.update(
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    bound_names.update(_grid_import_bindings(imports))
    unresolved = sorted(
        {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        - bound_names
        - set(_GRID_SAFE_BUILTINS)
    )
    if unresolved:
        raise LaunchContractError(f"Grid callable '{entry_point}' references unresolved names: {unresolved}")


def resolve_grid(
    grid: GridSpec,
    context: LaunchContext,
    repo_root: str | Path,
    *,
    meta: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Resolve and runtime-validate a literal or callable launch grid."""
    if isinstance(grid, tuple):
        return _validate_grid(grid)
    if isinstance(grid, list):
        return _validate_grid(tuple(grid))
    callable_grid = _load_grid_callable(grid, repo_root)
    try:
        resolved = callable_grid(context, meta)
    except Exception as exc:
        raise LaunchContractError(f"Grid callable '{grid}' failed: {exc}") from exc
    if not isinstance(resolved, (tuple, list)):
        raise LaunchContractError(f"Grid callable '{grid}' must return a tuple or list of integers")
    return _validate_grid(tuple(resolved))


def materialize_launch_arguments(
    launch: RawKernelLaunch | Mapping[str, Any],
    context: LaunchContext,
    *,
    triton_backend: Literal["nvt", "oait"] | None = None,
) -> dict[str, Any]:
    """Resolve every argument binding without imposing backend call syntax."""
    model = launch if isinstance(launch, RawKernelLaunch) else RawKernelLaunch.model_validate(launch)
    values: dict[str, Any] = {}
    for binding in model.arguments:
        if binding.target_triton_backends is not None:
            if triton_backend is None:
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' is compiler-scoped but no active Triton backend was provided"
                )
            if triton_backend not in binding.target_triton_backends:
                continue
        if binding.kind == "input":
            value = context.tensor(binding.name)  # type: ignore[arg-type]
            values[binding.parameter] = _apply_tensor_view(value, binding)
        elif binding.kind == "output":
            value = context.tensor(binding.name)  # type: ignore[arg-type]
            values[binding.parameter] = _apply_tensor_view(value, binding)
        elif binding.kind == "axis":
            try:
                values[binding.parameter] = context.axes[binding.name]  # type: ignore[index]
            except KeyError as exc:
                raise LaunchContractError(f"No runtime extent for axis '{binding.name}'") from exc
        elif binding.kind == "tensor_shape":
            values[binding.parameter] = context.shape(binding.name, binding.dimension)  # type: ignore[arg-type]
        elif binding.kind == "tensor_stride":
            values[binding.parameter] = context.stride(binding.name, binding.dimension)  # type: ignore[arg-type]
        elif binding.kind == "tensor_numel":
            values[binding.parameter] = context.numel(binding.name)  # type: ignore[arg-type]
        else:
            values[binding.parameter] = binding.value
    return values


def _apply_tensor_view(value: Any, binding: LaunchArgumentBinding) -> Any:
    """Apply an invocation-only, storage-sharing view to one tensor binding."""
    if binding.view is None:
        return value
    if binding.view == "flatten":
        reshape = getattr(value, "reshape", None)
        data_ptr = getattr(value, "data_ptr", None)
        is_contiguous = getattr(value, "is_contiguous", None)
        if not callable(reshape) or not callable(data_ptr) or not callable(is_contiguous):
            raise LaunchContractError(
                f"Binding '{binding.parameter}' cannot flatten a runtime value without tensor reshape/storage metadata"
            )
        try:
            if is_contiguous() is not True:
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' cannot flatten a non-contiguous runtime tensor"
                )
            flattened = reshape(-1)
            flattened_data_ptr = getattr(flattened, "data_ptr", None)
            if not callable(flattened_data_ptr) or flattened_data_ptr() != data_ptr():
                raise LaunchContractError(
                    f"Binding '{binding.parameter}' cannot create a storage-sharing flattened view"
                )
        except LaunchContractError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise LaunchContractError(
                f"Binding '{binding.parameter}' cannot create a storage-sharing flattened view"
            ) from exc
        return flattened
    raise LaunchContractError(f"Binding '{binding.parameter}' uses unsupported tensor view {binding.view!r}")


def materialize_launch(
    launch: RawKernelLaunch | Mapping[str, Any],
    context: LaunchContext,
    repo_root: str | Path,
    *,
    meta: Mapping[str, Any] | None = None,
    triton_backend: Literal["nvt", "oait"] | None = None,
) -> MaterializedLaunch:
    """Materialize grid and arguments for a later backend-specific adapter."""
    model = launch if isinstance(launch, RawKernelLaunch) else RawKernelLaunch.model_validate(launch)
    return MaterializedLaunch(
        grid=resolve_grid(model.grid, context, repo_root, meta=meta),
        arguments=materialize_launch_arguments(model, context, triton_backend=triton_backend),
    )


def _definition_data(definition: Any) -> Mapping[str, Any]:
    if isinstance(definition, Mapping):
        return definition
    model_dump = getattr(definition, "model_dump", None)
    if callable(model_dump):
        return _as_mapping(model_dump(mode="python"), "Definition")
    raise LaunchContractError("Definition must be a mapping or Pydantic model")


def _mapping_field(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _as_mapping(_field(data, name), f"Definition.{name}")


def _as_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise LaunchContractError(f"{description} must be a mapping")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise LaunchContractError(f"Missing required field '{name}'")
        return value[name]
    if not hasattr(value, name):
        raise LaunchContractError(f"Missing required field '{name}'")
    return getattr(value, name)


def _definition_constant_axes(definition: Mapping[str, Any]) -> dict[str, int]:
    constants: dict[str, int] = {}
    for name, axis in _mapping_field(definition, "axes").items():
        axis_type = _field(axis, "type")
        if axis_type == "const":
            constants[name] = int(_field(axis, "value"))
    return constants


def _merge_axis(axes: dict[str, int], name: str, extent: int, source: str) -> None:
    if name in axes and axes[name] != extent:
        raise LaunchContractError(f"Axis '{name}' is {axes[name]} but {source} resolved it to {extent}")
    axes[name] = extent


def _axis_extent(axis: Any, axes: Mapping[str, int], output_name: str) -> int:
    if not isinstance(axis, str) or axis not in axes:
        raise LaunchContractError(f"Definition output '{output_name}' uses unresolved axis '{axis}'")
    return int(axes[axis])


def _runtime_shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise LaunchContractError(f"Runtime value '{name}' does not expose a tensor shape")
    return tuple(int(extent) for extent in shape)


def _validate_json_literal(value: Any, path: str = "value") -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_literal(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _validate_json_literal(item, f"{path}.{key}")
        return
    raise ValueError(f"Literal binding {path} must contain only JSON-compatible values")


def _validate_grid(grid: tuple[Any, ...]) -> tuple[int, ...]:
    if not grid or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid):
        raise LaunchContractError("Launch grid must be a non-empty tuple of positive integers")
    return grid


def _entry_ast(entry_point: str, repo_root: str | Path) -> tuple[Path, str, ast.Module]:
    if entry_point.count("::") != 1:
        raise LaunchContractError(f"Invalid entry point '{entry_point}'; expected path.py::symbol")
    raw_path, symbol = entry_point.split("::", 1)
    if not symbol.isidentifier():
        raise LaunchContractError(f"Invalid entry-point symbol '{symbol}'")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or relative_path.suffix != ".py" or ".." in relative_path.parts:
        raise LaunchContractError(f"Entry-point path must be a repo-relative Python file: '{raw_path}'")
    root = Path(repo_root).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise LaunchContractError(f"Entry-point path escapes the repository: '{raw_path}'")
    if not path.is_file():
        raise LaunchContractError(f"Entry-point file does not exist: '{raw_path}'")
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise LaunchContractError(f"Entry-point file is not valid Python: '{raw_path}'") from exc
    return path, symbol, module


def _find_top_level_function(module: ast.Module, symbol: str, path: Path) -> ast.FunctionDef:
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == symbol]
    if len(matches) != 1:
        raise LaunchContractError(f"'{path}::{symbol}' must resolve to exactly one top-level function")
    return matches[0]


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


_GRID_IMPORT_ALLOWLIST = {"math", "operator"}
_GRID_SAFE_BUILTINS = {"abs": abs, "int": int, "len": len, "max": max, "min": min, "round": round}


def _validated_grid_imports(module: ast.Module, path: Path) -> list[ast.Import | ast.ImportFrom]:
    imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in imports:
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            raise LaunchContractError(f"Grid helper '{path}' cannot use wildcard imports")
        modules = (
            [alias.name.split(".", 1)[0] for alias in node.names] if isinstance(node, ast.Import) else [node.module]
        )
        if any(name not in _GRID_IMPORT_ALLOWLIST for name in modules):
            raise LaunchContractError(
                f"Grid helper '{path}' has an unrestricted import; only {sorted(_GRID_IMPORT_ALLOWLIST)} are allowed"
            )
    return imports


def _grid_import_bindings(imports: list[ast.Import | ast.ImportFrom]) -> set[str]:
    """Return names made available by validated module-level imports."""
    bindings: set[str] = set()
    for node in imports:
        if isinstance(node, ast.Import):
            bindings.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        else:
            bindings.update(alias.asname or alias.name for alias in node.names)
    return bindings


def _load_grid_callable(entry_point: str, repo_root: str | Path) -> Callable[..., Any]:
    # Grid helpers are reviewed repository code, not untrusted programs.  The
    # reduced namespace enforces a portable, deterministic callable contract;
    # process isolation remains the responsibility of the test/CI runner.
    inspect_grid_callable(entry_point, repo_root)
    path, symbol, module = _entry_ast(entry_point, repo_root)
    function = _find_top_level_function(module, symbol, path)
    imports = _validated_grid_imports(module, path)
    import_module = ast.Module(body=imports, type_ignores=[])
    function_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(import_module)
    ast.fix_missing_locations(function_module)
    namespace: dict[str, Any] = {"__builtins__": {"__import__": __import__}}
    exec(compile(import_module, str(path), "exec"), namespace)
    namespace["__builtins__"] = _GRID_SAFE_BUILTINS
    exec(compile(function_module, str(path), "exec"), namespace)
    return namespace[symbol]
