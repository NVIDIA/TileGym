# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Import inventory modules without initializing TileGym CUDA/Torch backends.
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory.generation import SourceRepository
    from tilegym.kernel_inventory.generation import TileGymDefinition
    from tilegym.kernel_inventory.generation import axis_var
    from tilegym.kernel_inventory.generation import definition_to_tilegym_json
    from tilegym.kernel_inventory.generation import make_definition
    from tilegym.kernel_inventory.generation import make_pinned_source_permalink
    from tilegym.kernel_inventory.generation import tensor
    from tilegym.kernel_inventory.generation import validate_definition_for_path
    from tilegym.kernel_inventory.generation import validate_tilegym_definition_model
    from tilegym.kernel_inventory.generation import write_definition_json
    from tilegym.kernel_inventory.source_contract import is_precise_source_permalink
    from tilegym.kernel_inventory.source_contract import validate_reference_source_contract
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
REFERENCE = (
    f"# Source: https://github.com/NVIDIA/TileGym/blob/{COMMIT}/src/tilegym/kernels/example.py#L10-L20\n"
    "import torch\n\n"
    "def run(input):\n"
    "    return input\n"
)


def _make_definition(*, include=()):
    return make_definition(
        name="wrapper",
        op_type="unit_test",
        axes={"M": axis_var()},
        inputs={"input": tensor(["M"], "float16")},
        outputs={"output": tensor(["M"], "float16")},
        reference=REFERENCE,
        include=include,
    )


def test_tilegym_definition_preserves_include_while_fib_view_is_flat():
    definition = _make_definition(include=["kernel_a", "_kernel_b"])

    assert isinstance(definition, TileGymDefinition)
    assert definition.include == ["kernel_a", "_kernel_b"]
    assert "include" not in definition.to_fib_definition().model_dump()
    assert definition_to_tilegym_json(definition)["include"] == ["kernel_a", "_kernel_b"]


def test_tilegym_definition_omits_empty_include_from_stable_json():
    assert "include" not in definition_to_tilegym_json(_make_definition())


def test_write_definition_json_round_trips_include(tmp_path):
    child = _make_definition()
    child.name = "kernel_a"
    (tmp_path / "kernel_a.json").write_text(json.dumps(definition_to_tilegym_json(child)), encoding="utf-8")
    path = tmp_path / "wrapper.json"
    wrapper = _make_definition(include=["kernel_a"])
    wrapper.reference = REFERENCE.replace("import torch\n", "import torch\nimport kernel_a\n").replace(
        "return input", "return kernel_a.run(input)"
    )
    write_definition_json(wrapper, path)

    checked_in_data = json.loads(path.read_text(encoding="utf-8"))
    assert checked_in_data["include"] == ["kernel_a"]
    assert validate_tilegym_definition_model(checked_in_data).include == ["kernel_a"]


@pytest.mark.parametrize("name", ["", " ", "kernel/a", "kernel.a", "class", "kernel-name"])
def test_tilegym_definition_rejects_nonlocal_include_names(name):
    with pytest.raises(ValueError, match="local Python module names"):
        _make_definition(include=[name])


@pytest.mark.parametrize("include", [["kernel_a", "kernel_a"], ["torch"], ["math"], ["run"]])
def test_tilegym_definition_rejects_duplicate_or_reserved_include_names(include):
    with pytest.raises(ValueError, match="unique|reserved"):
        _make_definition(include=include)


def test_path_dependent_generation_rejects_unresolved_include(tmp_path):
    wrapper = _make_definition(include=["missing"])
    wrapper.reference = REFERENCE.replace("import torch\n", "import torch\nimport missing\n").replace(
        "return input", "return missing.run(input)"
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_definition_for_path(wrapper, tmp_path / "wrapper.json")


def test_path_dependent_generation_rejects_malformed_hierarchical_public_path(tmp_path):
    path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/wrong.json"
    with pytest.raises(ValueError, match="named after its operation"):
        validate_definition_for_path(_make_definition(), path)


@pytest.mark.parametrize(
    "url",
    [
        f"https://github.com/NVIDIA/TileGym/blob/{COMMIT}/src/tilegym/kernels/example.py#L10-L20",
        f"https://huggingface.co/org/repo/blob/{COMMIT}/modeling.py#L3-L7",
    ],
)
def test_precise_source_permalink_accepts_supported_repositories(url):
    assert is_precise_source_permalink(url)
    validate_reference_source_contract(f"# Source: {url}\n\ndef run():\n    pass\n")


def test_make_pinned_source_permalink_uses_github_anchor():
    github_url = make_pinned_source_permalink(
        repo_kind=SourceRepository.TILEGYM_GITHUB,
        commit=COMMIT,
        path="src/tilegym/kernels/example.py",
        start_line=10,
        end_line=20,
    )

    assert github_url == (f"https://github.com/NVIDIA/TileGym/blob/{COMMIT}/src/tilegym/kernels/example.py#L10-L20")
    assert is_precise_source_permalink(github_url)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"repo_kind": "unknown"}, "Unsupported source repository"),
        ({"commit": "main"}, "40-hex"),
        ({"path": "../example.py"}, "repository-relative"),
        ({"start_line": 0}, "positive integer"),
        ({"start_line": 20, "end_line": 10}, "greater than or equal"),
    ],
)
def test_make_pinned_source_permalink_rejects_invalid_components(kwargs, match):
    arguments = {
        "repo_kind": SourceRepository.TILEGYM_GITHUB,
        "commit": COMMIT,
        "path": "src/tilegym/kernels/example.py",
        "start_line": 10,
        "end_line": 20,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_pinned_source_permalink(**arguments)
