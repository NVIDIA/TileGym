# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory.generation import make_solution
    from tilegym.kernel_inventory.generation import solution_to_tilegym_json
    from tilegym.kernel_inventory.generation import validate_tilegym_solution_model
    from tilegym.kernel_inventory.launch import LaunchArgumentBinding
    from tilegym.kernel_inventory.launch import LaunchContractError
    from tilegym.kernel_inventory.launch import RawKernelLaunch
    from tilegym.kernel_inventory.launch import allocate_definition_outputs
    from tilegym.kernel_inventory.launch import inspect_grid_callable
    from tilegym.kernel_inventory.launch import make_launch_context
    from tilegym.kernel_inventory.launch import materialize_launch
    from tilegym.kernel_inventory.launch import resolve_grid
    from tilegym.kernel_inventory.launch import validate_launch_contract
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)


class FakeTensor:
    def __init__(self, shape, *, fill=None):
        self.shape = tuple(shape)
        self.fill = fill

    def stride(self, dimension):
        strides = []
        stride = 1
        for extent in reversed(self.shape):
            strides.append(stride)
            stride *= extent
        return tuple(reversed(strides))[dimension]

    def numel(self):
        result = 1
        for extent in self.shape:
            result *= extent
        return result


class GridContext:
    def __init__(self, axes, inputs=None):
        self.axes = axes
        self.inputs = inputs or {}


@pytest.fixture
def definition():
    return {
        "name": "raw_add",
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": 8},
        },
        "inputs": {
            "x": {"shape": ["M", "N"], "dtype": "float16"},
            "scale": {"shape": None, "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["M", "N"], "dtype": "float16"},
            "checksum": {"shape": None, "dtype": "float32"},
        },
    }


@pytest.fixture
def raw_source(tmp_path):
    source = tmp_path / "kernels.py"
    source.write_text(
        "def raw_kernel(x, output, checksum, m, stride_m, n, elements, block=128):\n"
        "    pass\n\n"
        "def grid(context, meta=None):\n"
        "    return (context.axes['M'], meta['tiles'] if meta else 1)\n",
        encoding="utf-8",
    )
    return "kernels.py::raw_kernel", "kernels.py::grid"


@pytest.fixture
def launch(raw_source):
    raw_entry, grid_entry = raw_source
    del raw_entry
    return RawKernelLaunch(
        grid=grid_entry,
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "checksum", "kind": "output", "name": "checksum", "initialize": "zeros"},
            {"parameter": "m", "kind": "axis", "name": "M"},
            {"parameter": "stride_m", "kind": "tensor_stride", "name": "x", "dimension": 0},
            {"parameter": "n", "kind": "tensor_shape", "name": "x", "dimension": -1},
            {"parameter": "elements", "kind": "tensor_numel", "name": "x"},
            {"parameter": "block", "kind": "literal", "value": 64},
        ],
    )


def test_launch_contract_validates_definition_and_raw_ast_signature(tmp_path, definition, raw_source, launch):
    raw_entry, _ = raw_source
    signature = validate_launch_contract(launch, definition, raw_entry, tmp_path)

    assert [parameter.name for parameter in signature.parameters] == [
        "x",
        "output",
        "checksum",
        "m",
        "stride_m",
        "n",
        "elements",
        "block",
    ]
    assert signature.parameters[-1].required is False


def test_launch_contract_accepts_definition_input_as_output_seed(tmp_path, definition, raw_source, launch):
    raw_entry, _ = raw_source
    seeded = launch.model_copy(deep=True)
    seeded.arguments[1].initialize_from = "x"

    validate_launch_contract(seeded, definition, raw_entry, tmp_path)


def test_launch_contract_rejects_invalid_output_seed(tmp_path, definition, raw_source, launch):
    raw_entry, _ = raw_source
    seeded = launch.model_copy(deep=True)
    seeded.arguments[1].initialize_from = "missing"
    with pytest.raises(LaunchContractError, match="unknown Definition input"):
        validate_launch_contract(seeded, definition, raw_entry, tmp_path)

    seeded.arguments[1].initialize_from = "scale"
    with pytest.raises(LaunchContractError, match="identical shapes"):
        validate_launch_contract(seeded, definition, raw_entry, tmp_path)


@pytest.mark.parametrize("shape", [None, []])
def test_launch_contract_rejects_tensor_view_on_scalar_definition_tensor(tmp_path, shape):
    (tmp_path / "kernel.py").write_text("def kernel(scale):\n    pass\n", encoding="utf-8")
    definition = {
        "name": "raw_scalar",
        "axes": {},
        "inputs": {"scale": {"shape": shape, "dtype": "float32"}},
        "outputs": {},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[{"parameter": "scale", "kind": "input", "name": "scale", "view": "flatten"}],
    )

    with pytest.raises(LaunchContractError, match="scalar Definition tensor"):
        validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)


def test_output_seed_is_exclusive_to_uninitialized_output_bindings():
    with pytest.raises(ValidationError, match="cannot specify both"):
        LaunchArgumentBinding(
            parameter="output",
            kind="output",
            name="output",
            initialize="zeros",
            initialize_from="seed",
        )
    with pytest.raises(ValidationError, match="cannot specify an output seed"):
        LaunchArgumentBinding(parameter="input", kind="input", name="input", initialize_from="seed")


@pytest.mark.parametrize("view", ["reshape", "ravel", True, 1])
def test_binding_schema_rejects_unknown_tensor_views(view):
    with pytest.raises(ValidationError):
        LaunchArgumentBinding(parameter="input", kind="input", name="input", view=view)


@pytest.mark.parametrize("kind", ["axis", "tensor_shape", "tensor_stride", "tensor_numel", "literal"])
def test_binding_schema_limits_tensor_views_to_inputs_and_outputs(kind):
    data = {"parameter": "value", "kind": kind, "view": "flatten"}
    if kind != "literal":
        data["name"] = "value"
    if kind in {"tensor_shape", "tensor_stride"}:
        data["dimension"] = 0
    if kind == "literal":
        data["value"] = 1
    with pytest.raises(ValidationError, match="cannot specify a tensor view"):
        LaunchArgumentBinding.model_validate(data)


@pytest.mark.parametrize("targets", [[], ["oait", "oait"], ["oait", "nvt"], ["cuda"]])
def test_binding_schema_rejects_invalid_triton_backend_targets(targets):
    with pytest.raises(ValidationError):
        LaunchArgumentBinding(
            parameter="BLOCK",
            kind="literal",
            value=64,
            target_triton_backends=targets,
        )


def test_launch_contract_omits_triton_autotune_config_parameters(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.autotune(\n"
        "    configs=[triton.Config({'BLOCK': block}) for block in (32, 64)],\n"
        "    key=['n'],\n"
        ")\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BLOCK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "autotuned",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)
    block = next(parameter for parameter in signature.parameters if parameter.name == "BLOCK")
    assert block.provided_by_autotune
    assert not block.required

    conflicting = RawKernelLaunch(
        grid=[1],
        arguments=[*launch.arguments, {"parameter": "BLOCK", "kind": "literal", "value": 32}],
    )
    with pytest.raises(LaunchContractError, match="supplied by autotune"):
        validate_launch_contract(conflicting, definition, "kernel.py::kernel", tmp_path)


def test_launch_contract_requires_parameters_missing_from_a_reachable_autotune_config(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def is_nvt_backend():\n"
        "    return False\n\n"
        "def configs():\n"
        "    if is_nvt_backend():\n"
        "        return [triton.Config({'BK': 64, 'BV': 64}, num_warps=4)]\n"
        "    return [triton.Config({}, num_warps=4)]\n\n"
        "@triton.autotune(configs=configs(), key=['n', 'BK', 'BV'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr, BV: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "conditionally_autotuned",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
            {"parameter": "BV", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    tile_parameters = {
        parameter.name: parameter for parameter in signature.parameters if parameter.name in {"BK", "BV"}
    }
    assert set(tile_parameters) == {"BK", "BV"}
    assert all(parameter.required for parameter in tile_parameters.values())
    assert all(not parameter.provided_by_autotune for parameter in tile_parameters.values())
    assert all(parameter.possibly_provided_by_autotune for parameter in tile_parameters.values())


def test_launch_contract_accepts_compiler_scoped_binding_for_conditional_autotune_owner(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def is_nvt_backend():\n"
        "    return False\n\n"
        "def configs():\n"
        "    if is_nvt_backend():\n"
        "        return [triton.Config({'BK': 64}, num_warps=4)]\n"
        "    return [triton.Config({}, num_warps=4)]\n\n"
        "@triton.autotune(configs=configs(), key=['n'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "conditionally_autotuned",
        "axes": {"N": {"type": "var"}, "BK": {"type": "const", "value": 64}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {
                "parameter": "BK",
                "kind": "axis",
                "name": "BK",
                "target_triton_backends": ["oait"],
            },
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)
    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune
    assert block.possibly_provided_by_autotune

    context = make_launch_context(definition, {"x": FakeTensor((3,))}, outputs={"output": FakeTensor((3,))})
    assert materialize_launch(launch, context, tmp_path, triton_backend="oait").arguments["BK"] == 64
    assert "BK" not in materialize_launch(launch, context, tmp_path, triton_backend="nvt").arguments
    with pytest.raises(LaunchContractError, match="no active Triton backend"):
        materialize_launch(launch, context, tmp_path)


def test_launch_contract_rejects_compiler_scope_without_conditional_autotune_owner(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(x, output, n, BK):\n    pass\n", encoding="utf-8")
    definition = {
        "name": "not_autotuned",
        "axes": {"N": {"type": "var"}, "BK": {"type": "const", "value": 64}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {
                "parameter": "BK",
                "kind": "axis",
                "name": "BK",
                "target_triton_backends": ["oait"],
            },
        ],
    )

    with pytest.raises(LaunchContractError, match="some but not all reachable Triton autotune configs"):
        validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)


def test_launch_contract_requires_parameters_missing_from_a_conditional_config_mapping(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def is_nvt_backend():\n"
        "    return False\n\n"
        "def configs():\n"
        "    return [triton.Config({'BK': 64} if is_nvt_backend() else {}, num_warps=4)]\n\n"
        "@triton.autotune(configs=configs(), key=['n', 'BK'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "conditionally_mapped_autotune",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune


def test_launch_contract_resolves_unconditional_local_autotune_mapping(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def configs():\n"
        "    base = [({'BK': 128, 'BV': 128}, 8), ({'BK': 64, 'BV': 64}, 4)]\n"
        "    return [triton.Config(kw, num_warps=nw) for (kw, nw) in base]\n\n"
        "@triton.autotune(configs=configs(), key=['n'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr, BV: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "locally_mapped_autotune",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    tile_parameters = {
        parameter.name: parameter for parameter in signature.parameters if parameter.name in {"BK", "BV"}
    }
    assert set(tile_parameters) == {"BK", "BV"}
    assert all(not parameter.required for parameter in tile_parameters.values())
    assert all(parameter.provided_by_autotune for parameter in tile_parameters.values())


def test_launch_contract_treats_unknown_local_mapping_alternatives_as_unowned(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def external_base():\n"
        "    return [({}, 4)]\n\n"
        "def configs():\n"
        "    base = [({'BK': 64}, 4), *external_base()]\n"
        "    return [triton.Config(kw, num_warps=nw) for (kw, nw) in base]\n\n"
        "@triton.autotune(configs=configs(), key=['n', 'BK'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "partially_unknown_autotune",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune


@pytest.mark.parametrize(
    "configs_expression",
    [
        "[triton.Config({'BK': 64}), *EXTERNAL_CONFIGS]",
        "[triton.Config({'BK': 64})] + EXTERNAL_CONFIGS",
        "[triton.Config({'BK': 64})] if use_nvt() else EXTERNAL_CONFIGS",
    ],
)
def test_launch_contract_treats_unknown_config_sequence_alternatives_as_unowned(tmp_path, configs_expression):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "EXTERNAL_CONFIGS = get_external_configs()\n\n"
        "def use_nvt():\n"
        "    return False\n\n"
        f"@triton.autotune(configs={configs_expression}, key=['n', 'BK'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "partially_unknown_config_sequence",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune


def test_launch_contract_treats_mutated_local_mapping_as_unowned(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "def configs():\n"
        "    kw = {'BK': 64}\n"
        "    kw.pop('BK')\n"
        "    return [triton.Config(kw, num_warps=4)]\n\n"
        "@triton.autotune(configs=configs(), key=['n', 'BK'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "mutated_mapping_autotune",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune


@pytest.mark.parametrize(
    "mutation",
    [
        "cfgs.append(triton.Config({}))",
        "cfgs.extend(EXTERNAL_CONFIGS)",
        "cfgs += [triton.Config({})]",
    ],
)
def test_launch_contract_tracks_local_config_sequence_mutations(tmp_path, mutation):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "EXTERNAL_CONFIGS = get_external_configs()\n\n"
        "def configs():\n"
        "    cfgs = [triton.Config({'BK': 64})]\n"
        f"    {mutation}\n"
        "    return cfgs\n\n"
        "@triton.autotune(configs=configs(), key=['n', 'BK'])\n"
        "@triton.jit\n"
        "def kernel(x, output, n, BK: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "mutated_config_sequence",
        "axes": {"N": {"type": "var"}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float16"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "output", "kind": "output", "name": "output"},
            {"parameter": "n", "kind": "axis", "name": "N"},
            {"parameter": "BK", "kind": "literal", "value": 64},
        ],
    )

    signature = validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    block = next(parameter for parameter in signature.parameters if parameter.name == "BK")
    assert block.required
    assert not block.provided_by_autotune


def test_launch_contract_requires_all_or_none_triton_heuristic_parameters(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.heuristics({\n"
        "    'HAS_SCALE': lambda args: args['scale'] is not None,\n"
        "    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,\n"
        "})\n"
        "@triton.jit\n"
        "def kernel(x, output, scale, cu_seqlens, HAS_SCALE: tl.constexpr, IS_VARLEN: tl.constexpr):\n"
        "    pass\n",
        encoding="utf-8",
    )
    definition = {
        "name": "heuristic",
        "axes": {"N": {"type": "var"}},
        "inputs": {
            "x": {"shape": ["N"], "dtype": "float16"},
            "scale": {"shape": None, "dtype": "float32"},
            "cu_seqlens": {"shape": ["N"], "dtype": "int32"},
            "has_scale": {"shape": None, "dtype": "bool"},
            "is_varlen": {"shape": None, "dtype": "bool"},
        },
        "outputs": {"output": {"shape": ["N"], "dtype": "float16"}},
    }
    base_arguments = [
        {"parameter": "x", "kind": "input", "name": "x"},
        {"parameter": "output", "kind": "output", "name": "output"},
        {"parameter": "scale", "kind": "input", "name": "scale"},
        {"parameter": "cu_seqlens", "kind": "input", "name": "cu_seqlens"},
    ]
    decorator_launch = RawKernelLaunch(grid=[1], arguments=base_arguments)
    signature = validate_launch_contract(decorator_launch, definition, "kernel.py::kernel", tmp_path)
    assert {parameter.name for parameter in signature.parameters if parameter.provided_by_heuristic} == {
        "HAS_SCALE",
        "IS_VARLEN",
    }

    partial_launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            *base_arguments,
            {"parameter": "HAS_SCALE", "kind": "input", "name": "has_scale"},
        ],
    )
    with pytest.raises(LaunchContractError, match="every heuristic-owned parameter"):
        validate_launch_contract(partial_launch, definition, "kernel.py::kernel", tmp_path)

    explicit_launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            *base_arguments,
            {"parameter": "HAS_SCALE", "kind": "input", "name": "has_scale"},
            {"parameter": "IS_VARLEN", "kind": "input", "name": "is_varlen"},
        ],
    )
    validate_launch_contract(explicit_launch, definition, "kernel.py::kernel", tmp_path)


def test_definition_drives_output_allocation_and_launch_materialization(tmp_path, definition, launch):
    x = FakeTensor((3, 8))
    inputs = {"x": x, "scale": 0.5}
    context_without_outputs = make_launch_context(definition, inputs)
    allocations = []

    def allocator(*, name, shape, dtype, spec):
        allocations.append((name, shape, dtype, spec))
        return FakeTensor(shape)

    outputs = allocate_definition_outputs(definition, context_without_outputs.axes, allocator)
    context = make_launch_context(definition, inputs, outputs=outputs)
    materialized = materialize_launch(launch, context, tmp_path, meta={"tiles": 2})

    assert [(name, shape, dtype) for name, shape, dtype, _ in allocations] == [
        ("output", (3, 8), "float16"),
        ("checksum", (), "float32"),
    ]
    assert materialized.grid == (3, 2)
    assert materialized.arguments == {
        "x": x,
        "output": outputs["output"],
        "checksum": outputs["checksum"],
        "m": 3,
        "stride_m": 8,
        "n": 8,
        "elements": 24,
        "block": 64,
    }


def test_flatten_view_is_invocation_only_and_shares_tensor_storage():
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_shaped_copy",
        "axes": {"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 3}},
        "inputs": {"input": {"shape": ["M", "N"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["M", "N"], "dtype": "float32"}},
    }
    input_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    output_tensor = torch.zeros((2, 3), dtype=torch.float32)
    context = make_launch_context(definition, {"input": input_tensor}, outputs={"output": output_tensor})
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[
            {"parameter": "input", "kind": "input", "name": "input", "view": "flatten"},
            {"parameter": "output", "kind": "output", "name": "output", "view": "flatten"},
        ],
    )

    materialized = materialize_launch(launch, context, REPO_ROOT)

    assert input_tensor.shape == (2, 3)
    assert output_tensor.shape == (2, 3)
    assert materialized.arguments["input"].shape == (6,)
    assert materialized.arguments["output"].shape == (6,)
    assert materialized.arguments["input"].data_ptr() == input_tensor.data_ptr()
    assert materialized.arguments["output"].data_ptr() == output_tensor.data_ptr()
    materialized.arguments["output"].copy_(materialized.arguments["input"] + 1)
    torch.testing.assert_close(output_tensor, input_tensor + 1)


def test_flatten_view_rejects_noncontiguous_tensors_instead_of_copying():
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_noncontiguous",
        "axes": {"M": {"type": "const", "value": 3}, "N": {"type": "const", "value": 2}},
        "inputs": {"input": {"shape": ["M", "N"], "dtype": "float32"}},
        "outputs": {},
    }
    input_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    context = make_launch_context(definition, {"input": input_tensor})
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[{"parameter": "input", "kind": "input", "name": "input", "view": "flatten"}],
    )

    with pytest.raises(LaunchContractError, match="non-contiguous runtime tensor"):
        materialize_launch(launch, context, REPO_ROOT)


def test_flatten_view_rejects_noncontiguous_rank_one_tensor():
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_strided_vector",
        "axes": {"N": {"type": "const", "value": 4}},
        "inputs": {"input": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {},
    }
    input_tensor = torch.arange(8, dtype=torch.float32)[::2]
    assert input_tensor.shape == (4,)
    assert input_tensor.stride() == (2,)
    assert not input_tensor.is_contiguous()
    context = make_launch_context(definition, {"input": input_tensor})
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[{"parameter": "input", "kind": "input", "name": "input", "view": "flatten"}],
    )

    with pytest.raises(LaunchContractError, match="non-contiguous runtime tensor"):
        materialize_launch(launch, context, REPO_ROOT)


def test_flatten_view_rejects_runtime_values_without_tensor_view():
    definition = {
        "name": "raw_missing_view",
        "axes": {"M": {"type": "const", "value": 2}, "N": {"type": "const", "value": 3}},
        "inputs": {"input": {"shape": ["M", "N"], "dtype": "float32"}},
        "outputs": {},
    }
    context = make_launch_context(definition, {"input": FakeTensor((2, 3))})
    launch = RawKernelLaunch(
        grid=[1],
        arguments=[{"parameter": "input", "kind": "input", "name": "input", "view": "flatten"}],
    )

    with pytest.raises(LaunchContractError, match="without tensor reshape/storage metadata"):
        materialize_launch(launch, context, REPO_ROOT)


def test_literal_grid_accepts_json_list(definition):
    launch = RawKernelLaunch(grid=[2, 3], arguments=[])
    context = make_launch_context(definition, {"x": FakeTensor((1, 8)), "scale": 1.0})

    assert launch.grid == (2, 3)
    assert resolve_grid(launch.grid, context, REPO_ROOT) == (2, 3)


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"BLOCK_SIZE": 16}, (5, 2)),
        (None, (3, 2)),
    ],
)
def test_callable_grid_uses_context_and_meta_or_axis_fallback(tmp_path, meta, expected):
    grid_source = tmp_path / "grid.py"
    grid_source.write_text(
        "def grid(context, meta=None):\n"
        "    block_size = context.axes['BLOCK_SIZE'] if meta is None else meta['BLOCK_SIZE']\n"
        "    return ((context.axes['N'] + block_size - 1) // block_size, context.inputs['batches'])\n",
        encoding="utf-8",
    )
    context = GridContext({"N": 65, "BLOCK_SIZE": 32}, {"batches": 2})

    assert resolve_grid("grid.py::grid", context, tmp_path, meta=meta) == expected


@pytest.mark.parametrize(
    ("return_value", "match"),
    [
        ("1", "must return a tuple or list of integers"),
        ("(0,)", "positive integers"),
        ("(True,)", "positive integers"),
    ],
)
def test_callable_grid_validates_resolved_values(tmp_path, return_value, match):
    grid_source = tmp_path / "grid.py"
    grid_source.write_text(
        f"def grid(context, meta=None):\n    return {return_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(LaunchContractError, match=match):
        resolve_grid("grid.py::grid", GridContext({}), tmp_path)


@pytest.mark.parametrize("grid", [[], [0], [-1, 2], [True]])
def test_literal_grid_requires_positive_integers(grid):
    with pytest.raises((LaunchContractError, ValidationError), match="positive integers"):
        RawKernelLaunch(grid=grid, arguments=[])


def test_binding_schema_rejects_duplicated_tensor_metadata():
    with pytest.raises(ValidationError, match="shape"):
        LaunchArgumentBinding.model_validate({"parameter": "x", "kind": "input", "name": "x", "shape": ["M", "N"]})


def test_launch_contract_rejects_duplicate_output_destination(tmp_path, definition, raw_source, launch):
    launch.arguments.append(
        LaunchArgumentBinding(
            parameter="block",
            kind="output",
            name="output",
        )
    )
    raw_entry, _ = raw_source
    with pytest.raises(LaunchContractError, match="duplicate raw destination"):
        validate_launch_contract(launch, definition, raw_entry, tmp_path)


def test_launch_contract_rejects_unknown_definition_reference(tmp_path, definition, raw_source, launch):
    raw_entry, _ = raw_source
    launch.arguments[0] = LaunchArgumentBinding(parameter="x", kind="input", name="missing")

    with pytest.raises(LaunchContractError, match="unknown Definition input"):
        validate_launch_contract(launch, definition, raw_entry, tmp_path)


def test_launch_contract_rejects_missing_and_unknown_raw_parameters(tmp_path, definition):
    definition = dict(definition)
    definition["outputs"] = {}
    (tmp_path / "kernel.py").write_text("def kernel(x, required):\n    pass\n", encoding="utf-8")
    launch = RawKernelLaunch(
        grid=(1,),
        arguments=[
            {"parameter": "x", "kind": "input", "name": "x"},
            {"parameter": "not_a_parameter", "kind": "literal", "value": 1},
        ],
    )

    with pytest.raises(LaunchContractError, match="unknown raw entry parameters"):
        validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)

    launch.arguments.pop()
    with pytest.raises(LaunchContractError, match="Required raw entry parameters"):
        validate_launch_contract(launch, definition, "kernel.py::kernel", tmp_path)


@pytest.mark.parametrize(
    "entry_point",
    ["../outside.py::grid", "/tmp/outside.py::grid", "grid.py::(lambda context: (1,))"],
)
def test_grid_callable_rejects_escaping_paths_and_embedded_expressions(tmp_path, entry_point):
    with pytest.raises(LaunchContractError):
        inspect_grid_callable(entry_point, tmp_path)


def test_grid_callable_rejects_wrong_signature_nested_callable_and_unrestricted_import(tmp_path):
    wrong_signature = tmp_path / "wrong_signature.py"
    wrong_signature.write_text("def grid(meta):\n    return (1,)\n", encoding="utf-8")
    with pytest.raises(LaunchContractError, match="exact signature"):
        inspect_grid_callable("wrong_signature.py::grid", tmp_path)

    nested = tmp_path / "nested.py"
    nested.write_text(
        "def grid(context, meta=None):\n    helper = lambda value: value\n    return (helper(1),)\n",
        encoding="utf-8",
    )
    with pytest.raises(LaunchContractError, match="lambdas or nested functions"):
        inspect_grid_callable("nested.py::grid", tmp_path)

    imported = tmp_path / "imported.py"
    imported.write_text("import os\n\ndef grid(context, meta=None):\n    return (1,)\n", encoding="utf-8")
    with pytest.raises(LaunchContractError, match="unrestricted import"):
        inspect_grid_callable("imported.py::grid", tmp_path)


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("BLOCK = 128\n\ndef grid(context, meta=None):\n    return (BLOCK,)\n", "BLOCK"),
        ("def helper():\n    return 1\n\ndef grid(context, meta=None):\n    return (helper(),)\n", "helper"),
        ("def grid(context, meta=None):\n    return (sum(context.axes.values()),)\n", "sum"),
    ],
)
def test_grid_callable_rejects_names_missing_from_runtime_namespace(tmp_path, source, name):
    path = tmp_path / "unresolved.py"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(LaunchContractError, match=rf"unresolved names.*{name}"):
        inspect_grid_callable("unresolved.py::grid", tmp_path)


def test_grid_callable_accepts_validated_imports_safe_builtins_and_locals(tmp_path):
    path = tmp_path / "valid.py"
    path.write_text(
        "import math as m\n\ndef grid(context, meta=None):\n"
        "    width = max(1, int(m.ceil(context.axes['N'] / 8)))\n"
        "    return (width,)\n",
        encoding="utf-8",
    )

    inspect_grid_callable("valid.py::grid", tmp_path)
    assert resolve_grid("valid.py::grid", GridContext({"N": 17}), tmp_path) == (3,)


def test_context_rejects_axis_conflicts_and_output_allocator_rejects_unresolved_axes(definition):
    with pytest.raises(LaunchContractError, match="Axis 'N'"):
        make_launch_context(definition, {"x": FakeTensor((2, 7)), "scale": 1.0})

    with pytest.raises(LaunchContractError, match="unresolved axis"):
        allocate_definition_outputs(definition, {"N": 8}, lambda **kwargs: kwargs)


def test_solution_preserves_launch_while_transient_fib_view_omits_it(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(input, output):\n    pass\n", encoding="utf-8")
    launch = {
        "grid": [1],
        "arguments": [
            {"parameter": "input", "kind": "input", "name": "input"},
            {
                "parameter": "output",
                "kind": "output",
                "name": "output",
                "initialize_from": "input",
                "view": "flatten",
            },
        ],
    }
    solution = make_solution(
        name="copy_triton",
        definition="copy",
        author="test",
        spec={
            "language": "triton",
            "target_hardware": ["SM100"],
            "entry_point": "kernel.py::kernel",
            "dependencies": ["torch", "triton"],
            "destination_passing_style": True,
        },
        sources="kernel.py",
        repo_root=tmp_path,
        launch=launch,
    )

    assert solution.launch is not None
    assert solution.to_tilegym_dict()["launch"]["grid"] == [1]
    assert solution.to_tilegym_dict()["launch"]["arguments"][1]["initialize_from"] == "input"
    assert solution.to_tilegym_dict()["launch"]["arguments"][1]["view"] == "flatten"
    assert not hasattr(solution.to_fib_solution(tmp_path), "launch")


def test_solution_rejects_compiler_scoped_binding_for_non_triton_language(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(input, output, BLOCK):\n    pass\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="valid only for language='triton'"):
        make_solution(
            name="copy_cutile",
            definition="copy",
            author="test",
            spec={
                "language": "cuda-tile",
                "target_hardware": ["SM100"],
                "entry_point": "kernel.py::kernel",
                "dependencies": ["torch", "cuda.tile"],
                "destination_passing_style": True,
            },
            sources="kernel.py",
            repo_root=tmp_path,
            launch={
                "grid": [1],
                "arguments": [
                    {
                        "parameter": "BLOCK",
                        "kind": "literal",
                        "value": 64,
                        "target_triton_backends": ["oait"],
                    }
                ],
            },
        )


def test_solution_serialization_preserves_explicit_null_literal(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(optional=None):\n    pass\n", encoding="utf-8")
    solution = make_solution(
        name="optional_triton",
        definition="optional",
        author="test",
        spec={
            "language": "triton",
            "target_hardware": ["SM100"],
            "entry_point": "kernel.py::kernel",
            "dependencies": ["torch", "triton"],
            "destination_passing_style": True,
        },
        sources="kernel.py",
        repo_root=tmp_path,
        launch={
            "grid": [1],
            "arguments": [{"parameter": "optional", "kind": "literal", "value": None}],
        },
    )

    serialized = solution_to_tilegym_json(solution)
    assert serialized["launch"]["arguments"][0]["value"] is None
    validate_tilegym_solution_model(serialized, tmp_path)
