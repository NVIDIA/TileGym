# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from pathlib import PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import inventory modules without running tilegym/__init__.py. The top-level
# package initializes CUDA/Torch backends, which is unrelated to JSON schema checks.
tilegym_pkg = types.ModuleType("tilegym")
tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
sys.modules.setdefault("tilegym", tilegym_pkg)
transformers_pkg = types.ModuleType("tilegym.transformers")
transformers_pkg.__path__ = [str(REPO_ROOT / "src/tilegym/transformers")]
sys.modules.setdefault("tilegym.transformers", transformers_pkg)

from tilegym.transformers.inventory_common import SourceContractError
from tilegym.transformers.inventory_common import validate_reference_source_contract
from tilegym.transformers.kernel_inventory import KernelInventoryError
from tilegym.transformers.kernel_inventory import iter_kernel_definition_paths
from tilegym.transformers.kernel_inventory import iter_kernel_python_paths
from tilegym.transformers.kernel_inventory import iter_kernel_solution_paths
from tilegym.transformers.kernel_inventory import load_json
from tilegym.transformers.kernel_inventory import materialize_solution_sources
from tilegym.transformers.kernel_inventory import normalize_solution_source_paths
from tilegym.transformers.kernel_inventory import validate_definition
from tilegym.transformers.kernel_inventory import validate_solution
from tilegym.transformers.kernel_inventory import validate_solution_entry_point

FLOAT32_TENSOR_ALLOWLIST = {
    ("qwen3_5_gdr_preprocess", "inputs", "A_log"),
    ("qwen3_5_gdr_preprocess", "inputs", "dt_bias"),
    ("qwen3_5_gdr_preprocess", "outputs", "g"),
}


def _definition():
    return {
        "name": "rmsnorm_d128",
        "description": "RMSNorm over a fixed hidden size.",
        "op_type": "rmsnorm",
        "tags": ["status:draft", "model:test"],
        "axes": {
            "M": {"type": "var"},
            "D": {"type": "const", "value": 128},
        },
        "inputs": {
            "input": {"shape": ["M", "D"], "dtype": "float16"},
            "weight": {"shape": ["D"], "dtype": "float16"},
            "eps": {"shape": None, "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["M", "D"], "dtype": "float16"},
        },
        "constraints": ["D == 128"],
        "reference": (
            "# Source: https://github.com/huggingface/transformers/blob/"
            "0123456789abcdef0123456789abcdef01234567/src/transformers/models/test/modeling_test.py#L1-L2\n"
            "import torch\n\n"
            "def run(input, weight, eps):\n"
            "    return input * weight"
        ),
    }


def _solution():
    return {
        "name": "rmsnorm_d128_cutile",
        "definition": "rmsnorm_d128",
        "author": "tilegym-agent",
        "spec": {
            "language": "cuda-tile",
            "target_hardware": ["NVIDIA_B200"],
            "entry_point": "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py::run",
            "destination_passing_style": False,
            "dependencies": ["torch", "cuda-tile"],
        },
        "sources": {
            "path": ["src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"],
        },
    }


def _is_transformer_kernel_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) >= 6
        and parts[0] == "src"
        and parts[1] == "tilegym"
        and parts[2] == "transformers"
        and parts[4] == "kernels"
    )


def _assert_definition_reference_contract(path: Path, definition: dict):
    reference = definition["reference"]
    try:
        validate_reference_source_contract(reference)
    except SourceContractError as exc:
        raise AssertionError(f"{path}: {exc}") from exc

    try:
        module_ast = ast.parse(reference, filename=str(path))
    except SyntaxError as exc:
        raise AssertionError(f"{path}: Definition.reference must be valid Python") from exc

    run_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
    ]
    assert len(run_nodes) == 1, f"{path}: Definition.reference must define exactly one global run function"

    run_args = run_nodes[0].args
    assert run_args.vararg is None and run_args.kwarg is None, (
        f"{path}: Definition.reference run function must not use *args or **kwargs"
    )
    arg_names = [arg.arg for arg in (*run_args.posonlyargs, *run_args.args, *run_args.kwonlyargs)]
    assert arg_names == list(definition["inputs"]), (
        f"{path}: Definition.reference run arguments {arg_names} must match Definition.inputs "
        f"{list(definition['inputs'])}"
    )


def _duplicate_names(entries: list[tuple[str, Path]]) -> list[str]:
    seen = set()
    duplicates = set()
    for name, _path in entries:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)


def test_validate_definition_accepts_flashinfer_shape():
    validate_definition(_definition())


def test_validate_definition_requires_reference_run():
    definition = _definition()
    definition["reference"] = "# def run(input, weight, eps):\n"
    with pytest.raises(KernelInventoryError, match="top-level function|global run function"):
        validate_definition(definition)


def test_validate_definition_rejects_malformed_reference():
    definition = _definition()
    definition["reference"] = "def run(input, weight, eps):\n    return ("
    with pytest.raises(KernelInventoryError, match="valid Python"):
        validate_definition(definition)


def test_validate_definition_rejects_run_prefix_only():
    definition = _definition()
    definition["reference"] = "def run2(input, weight, eps):\n    return input\n"
    with pytest.raises(KernelInventoryError, match="top-level function|global run function"):
        validate_definition(definition)


def test_validate_definition_rejects_undefined_axis():
    definition = _definition()
    definition["inputs"]["input"]["shape"] = ["M", "Missing"]
    with pytest.raises(KernelInventoryError, match="undefined"):
        validate_definition(definition)


def test_validate_definition_rejects_overlapping_input_output_names():
    definition = _definition()
    definition["outputs"] = {
        "input": {"shape": ["M", "D"], "dtype": "float16"},
    }
    with pytest.raises(KernelInventoryError, match="overlap"):
        validate_definition(definition)


def test_validate_definition_rejects_bad_constraint_syntax():
    definition = _definition()
    definition["constraints"] = ["D =="]
    with pytest.raises(KernelInventoryError, match="Constraints"):
        validate_definition(definition)


def test_validate_definition_requires_precise_source_permalink():
    definition = _definition()
    definition["reference"] = "import torch\n\ndef run(input, weight, eps):\n    return input * weight\n"
    with pytest.raises(KernelInventoryError, match="# Source:"):
        validate_definition(definition)


def test_validate_definition_requires_run_args_to_match_inputs():
    definition = _definition()
    definition["reference"] = (
        "# Source: https://github.com/huggingface/transformers/blob/"
        "0123456789abcdef0123456789abcdef01234567/src/transformers/models/test/modeling_test.py#L1-L2\n"
        "import torch\n\n"
        "def run(weight, input, eps):\n"
        "    return input * weight"
    )
    with pytest.raises(KernelInventoryError, match="must match Definition.inputs"):
        validate_definition(definition)


def test_validate_solution_accepts_path_only_sources(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    validate_solution(solution, repo_root=tmp_path)
    validate_solution_entry_point(solution, repo_root=tmp_path)
    assert normalize_solution_source_paths(solution) == ["src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"]


def test_validate_solution_accepts_file_object_sources_without_content(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    solution["sources"] = [
        {
            "path": "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py",
        }
    ]
    validate_solution(solution, repo_root=tmp_path)


def test_materialize_solution_sources_adds_content(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    materialized = materialize_solution_sources(_solution(), repo_root=tmp_path)
    assert materialized["sources"] == [
        {
            "path": "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py",
            "content": "def run(input, weight, eps):\n    return input\n",
        }
    ]


def test_materialize_solution_for_fib_accepts_path_only_sources(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    from tilegym.transformers.inventory_generation import materialize_solution_for_fib

    fib_solution = materialize_solution_for_fib(_solution(), repo_root=tmp_path)
    assert fib_solution.spec.language == "python"
    assert fib_solution.sources[0].path == "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    assert fib_solution.sources[0].content == "def run(input, weight, eps):\n    return input\n"


def test_make_solution_accepts_single_source_string(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    from tilegym.transformers.inventory_generation import make_solution

    source = "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    solution = make_solution(
        name="rmsnorm_d128_cutile",
        definition="rmsnorm_d128",
        author="tilegym-agent",
        spec=_solution()["spec"],
        sources=source,
        repo_root=tmp_path,
    )

    assert [entry.path for entry in solution.sources] == [source]


def test_solution_rejects_missing_source_path(tmp_path):
    with pytest.raises(KernelInventoryError, match="does not exist"):
        validate_solution(_solution(), repo_root=tmp_path)


def test_solution_rejects_source_path_outside_repo(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    solution["sources"] = {"path": ["../outside.py"]}
    with pytest.raises(KernelInventoryError, match="repo-relative"):
        validate_solution(solution, repo_root=tmp_path)

    solution = _solution()
    solution["sources"] = {"path": [str(source_path.resolve())]}
    with pytest.raises(KernelInventoryError, match="repo-relative"):
        validate_solution(solution, repo_root=tmp_path)


def test_solution_rejects_entry_point_outside_repo(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    solution["spec"]["entry_point"] = "../outside.py::run"
    with pytest.raises(KernelInventoryError, match="repo-relative"):
        validate_solution_entry_point(solution, repo_root=tmp_path)


def test_solution_rejects_duplicate_source_paths(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def run(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    solution["sources"] = {
        "path": [
            "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py",
            "src/tilegym/transformers/test_model/kernels/rmsnorm_d128.py",
        ]
    }
    with pytest.raises(KernelInventoryError, match="Duplicate source path"):
        validate_solution(solution, repo_root=tmp_path)


def test_solution_rejects_entry_file_missing_from_sources(tmp_path):
    source_path = tmp_path / "src/tilegym/transformers/test_model/kernels/other.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def other(input, weight, eps):\n    return input\n", encoding="utf-8")

    solution = _solution()
    solution["sources"] = {"path": ["src/tilegym/transformers/test_model/kernels/other.py"]}
    with pytest.raises(KernelInventoryError, match="Entry source file"):
        validate_solution(solution, repo_root=tmp_path)


def test_sample_metadata_is_json_serializable():
    json.dumps(_definition())
    json.dumps(_solution())


def test_all_current_kernel_definitions_validate():
    for path in iter_kernel_definition_paths(REPO_ROOT):
        definition = load_json(path)
        assert path.stem == definition["name"], f"{path}: Definition filename must match Definition.name"
        _assert_tensor_dtypes_match_model_precision(path, definition)
        validate_definition(definition)
        _assert_definition_reference_contract(path, definition)


def test_all_current_kernel_solutions_validate():
    for path in iter_kernel_solution_paths(REPO_ROOT):
        solution = load_json(path)
        assert path.stem == solution["definition"], f"{path}: Solution filename must match Solution.definition"
        assert solution["name"].startswith(f"{solution['definition']}_"), (
            f"{path}: Solution.name must be derived from Solution.definition"
        )
        validate_solution(solution, repo_root=REPO_ROOT)
        validate_solution_entry_point(solution, repo_root=REPO_ROOT)


def test_kernel_definition_solution_catalog_is_complete():
    definition_entries = [(load_json(path)["name"], path) for path in iter_kernel_definition_paths(REPO_ROOT)]
    solution_entries = [(load_json(path)["name"], path) for path in iter_kernel_solution_paths(REPO_ROOT)]

    duplicate_definitions = _duplicate_names(definition_entries)
    duplicate_solutions = _duplicate_names(solution_entries)
    assert not duplicate_definitions, f"Duplicate Definition names: {duplicate_definitions}"
    assert not duplicate_solutions, f"Duplicate Solution names: {duplicate_solutions}"

    definitions = dict(definition_entries)
    solutions = {name: load_json(path) for name, path in solution_entries}
    definition_names = set(definitions)
    solution_definitions = {solution["definition"] for solution in solutions.values()}

    missing_definitions = sorted(solution_definitions - definition_names)
    missing_solutions = sorted(definition_names - solution_definitions)
    assert not missing_definitions, f"Solutions reference missing Definitions: {missing_definitions}"
    assert not missing_solutions, f"Definitions without a matching Solution: {missing_solutions}"


def test_kernel_solution_sources_stay_in_dedicated_kernel_modules():
    for path in iter_kernel_solution_paths(REPO_ROOT):
        solution = load_json(path)
        source_paths = normalize_solution_source_paths(solution)
        entry_path = solution["spec"]["entry_point"].split("::", 1)[0]

        assert entry_path in source_paths, f"{path}: Solution entry point file must be listed in sources.path"
        assert _is_transformer_kernel_path(entry_path), (
            f"{path}: Solution entry point must live under src/tilegym/transformers/<module>/kernels/"
        )
        for source_path in source_paths:
            assert _is_transformer_kernel_path(source_path), (
                f"{path}: Solution source path must live under src/tilegym/transformers/<module>/kernels/"
            )


def _assert_tensor_dtypes_match_model_precision(path: Path, definition: dict):
    definition_name = definition["name"]
    for section in ("inputs", "outputs"):
        for name, spec in definition[section].items():
            if spec["shape"] is None or spec["dtype"] != "float32":
                continue
            key = (definition_name, section, name)
            assert key in FLOAT32_TENSOR_ALLOWLIST, (
                f"{path}: tensor {section}.{name} uses float32. Model activations/weights should keep the "
                "model dtype unless the upstream model/kernel semantics require float32; add a narrow "
                "allowlist entry for verified float32 tensors."
            )


def test_all_dedicated_kernel_modules_import():
    for path in iter_kernel_python_paths(REPO_ROOT):
        module_name = f"_tilegym_transformer_kernel_{path.stem}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
