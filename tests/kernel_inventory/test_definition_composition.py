# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory.composition import DefinitionCompositionError
    from tilegym.kernel_inventory.composition import installed_reference_modules
    from tilegym.kernel_inventory.composition import validate_definition_composition
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)


def _definition(name, reference, include=None):
    definition = {
        "name": name,
        "op_type": "test",
        "axes": {},
        "inputs": {},
        "outputs": {},
        "reference": reference,
    }
    if include is not None:
        definition["include"] = include
    return definition


def _write(path, definition):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


def test_wrapper_composition_validates_and_loads_leaf_modules(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value + 1\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value):\n    output = leaf.run(value)\n    return output\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)

    validate_definition_composition(wrapper, wrapper_path)
    with installed_reference_modules(wrapper, wrapper_path):
        namespace = {}
        exec(wrapper["reference"], namespace)
        assert namespace["run"](4) == 5
    assert "leaf" not in sys.modules


def test_wrapper_composition_rejects_missing_include_call(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition("op", "import leaf\n\ndef run(value):\n    return value\n", ["leaf"])
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)

    with pytest.raises(DefinitionCompositionError, match="not called"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_checks_leaf_signature(tmp_path):
    leaf = _definition("leaf", "def run(value, scale):\n    return value * scale\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value):\n    return leaf.run(value)\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)

    with pytest.raises(DefinitionCompositionError, match="missing required"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_rejects_include_cycles(tmp_path):
    first = _definition("first", "import second\n\ndef run(value):\n    return second.run(value)\n", ["second"])
    second = _definition("second", "import first\n\ndef run(value):\n    return first.run(value)\n", ["first"])
    first_path = _write(tmp_path / "first.json", first)
    _write(tmp_path / "second.json", second)

    with pytest.raises(DefinitionCompositionError, match="cycle"):
        validate_definition_composition(first, first_path)


def test_wrapper_composition_rejects_top_level_helpers(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef helper(value):\n    return value\n\ndef run(value):\n    return leaf.run(helper(value))\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)

    with pytest.raises(DefinitionCompositionError, match="only imports"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_rejects_nested_class_body_include_escape(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        (
            "import leaf\n\n"
            "def run(value):\n"
            "    class Hidden:\n"
            "        if False:\n"
            "            leaf.run(value)\n"
            "    return value\n"
        ),
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="must not define nested functions or classes"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_accepts_supported_torch_grad_context(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        ("import torch\nimport leaf\n\ndef run(value):\n    with torch.no_grad():\n        return leaf.run(value)\n"),
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_accepts_valid_constant_bitwise_expressions(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        (
            "import leaf\n\n"
            "def run(value):\n"
            "    shifted = 1 << 3\n"
            "    masked = shifted & 7\n"
            "    inverted = ~masked\n"
            "    return leaf.run(value)\n"
        ),
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_static_validation_stops_at_structural_contract(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        (
            "import math\n"
            "import leaf\n\n"
            "def run(value):\n"
            "    target = None\n"
            "    target[0] = 1\n"
            "    unused = math.sqrt(-1)\n"
            "    with None:\n"
            "        pass\n"
            "    if False:\n"
            "        return leaf.run(value)\n"
            "    return value\n"
        ),
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    validate_definition_composition(wrapper, wrapper_path)


def test_reference_modules_load_nested_includes_in_dependency_order(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value + 1\n")
    middle = _definition(
        "middle",
        "import leaf\n\ndef run(value):\n    return leaf.run(value) * 2\n",
        ["leaf"],
    )
    wrapper = _definition(
        "op",
        "import middle\n\ndef run(value):\n    return middle.run(value)\n",
        ["middle"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "middle.json", middle)
    _write(tmp_path / "leaf.json", leaf)

    validate_definition_composition(wrapper, wrapper_path)
    with installed_reference_modules(wrapper, wrapper_path):
        namespace = {}
        exec(wrapper["reference"], namespace)
        assert namespace["run"](4) == 10


def _hierarchical_path(tmp_path, level, name="op"):
    root = tmp_path / "src/tilegym/suites/example/kernel_definitions/op"
    if level == "public":
        return root / "op.json"
    return root / "cutile" / f"{name}.json"


def test_hierarchical_executable_wrapper_requires_nonempty_include(tmp_path):
    wrapper = _definition("op", "def run(value):\n    return value\n")
    with pytest.raises(DefinitionCompositionError, match="must include at least one leaf"):
        validate_definition_composition(wrapper, _hierarchical_path(tmp_path, "wrapper"))


def test_hierarchical_unsupported_wrapper_allows_empty_include_but_stays_flat(tmp_path):
    wrapper = _definition("op", "def run(**kwargs):\n    raise NotImplementedError('deprecated')\n")
    wrapper["tags"] = ["runtime:unsupported"]
    validate_definition_composition(wrapper, _hierarchical_path(tmp_path, "wrapper"))

    wrapper["reference"] = "def run(**kwargs):\n    import os\n    raise NotImplementedError('deprecated')\n"
    with pytest.raises(DefinitionCompositionError, match="must not contain imports"):
        validate_definition_composition(wrapper, _hierarchical_path(tmp_path, "wrapper"))


@pytest.mark.parametrize("level", ["public", "leaf"])
@pytest.mark.parametrize("include", [[], ["child"]])
def test_hierarchical_public_and_leaf_definitions_reject_include(tmp_path, level, include):
    definition = _definition("op" if level == "public" else "leaf", "def run(value):\n    return value\n", include)
    with pytest.raises(DefinitionCompositionError, match=f"{level} Definitions must not declare include"):
        validate_definition_composition(definition, _hierarchical_path(tmp_path, level, "leaf"))


def test_hierarchical_wrapper_rejects_leaf_with_nested_include(tmp_path):
    wrapper = _definition("op", "import leaf\n\ndef run(value):\n    return leaf.run(value)\n", ["leaf"])
    leaf = _definition("leaf", "def run(value):\n    return value\n", ["nested"])
    wrapper_path = _write(_hierarchical_path(tmp_path, "wrapper"), wrapper)
    _write(wrapper_path.parent / "leaf.json", leaf)
    _write(wrapper_path.parent / "nested.json", _definition("nested", "def run(value):\n    return value\n"))
    with pytest.raises(DefinitionCompositionError, match="leaf Definitions must not declare include"):
        validate_definition_composition(wrapper, wrapper_path)


def test_hierarchical_wrapper_rejects_leaf_with_explicit_empty_include(tmp_path):
    wrapper = _definition("op", "import leaf\n\ndef run(value):\n    return leaf.run(value)\n", ["leaf"])
    leaf = _definition("leaf", "def run(value):\n    return value\n", [])
    wrapper_path = _write(_hierarchical_path(tmp_path, "wrapper"), wrapper)
    _write(wrapper_path.parent / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="leaf Definitions must not declare include"):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("import os\n    return leaf.run(value)", "must not contain imports"),
        ("return unknown.api(value)", "unsupported attribute root"),
        ("return leaf.helper(value)", "may only be referenced"),
    ],
)
def test_wrapper_composition_rejects_flatness_escapes(tmp_path, body, match):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition("op", f"import leaf\n\ndef run(value):\n    {body}\n", ["leaf"])
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match=match):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("checked = leaf.run(value)\n    torch = leaf\n    return torch.helper(value)", "must not shadow"),
        ("checked = leaf.run(value)\n    alias = leaf\n    return alias.helper(value)", "may only be referenced"),
    ],
)
def test_wrapper_composition_rejects_include_alias_and_import_shadowing(tmp_path, body, match):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition("op", f"import torch\nimport leaf\n\ndef run(value):\n    {body}\n", ["leaf"])
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match=match):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    "shadow_import",
    ["import torch as leaf", "from torch import ops as leaf"],
)
def test_wrapper_composition_rejects_top_level_import_alias_over_include(tmp_path, shadow_import):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        f"import leaf\n{shadow_import}\n\ndef run(value):\n    return leaf.run(value)\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="shadows included Definition"):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    "reference",
    [
        "import run\n\ndef run(value):\n    return run.run(value)\n",
        "import leaf\nimport torch as run\n\ndef run(value):\n    checked = leaf.run(value)\n    return run.sin(checked)\n",
    ],
)
def test_wrapper_composition_reserves_global_run_binding(tmp_path, reference):
    includes = ["run"] if "import run\n" in reference else ["leaf"]
    wrapper = _definition("op", reference, includes)
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    for include in includes:
        _write(tmp_path / f"{include}.json", _definition(include, "def run(value):\n    return value\n"))
    with pytest.raises(DefinitionCompositionError, match="reserved|global run function"):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize("control", ["break", "continue"])
def test_wrapper_composition_rejects_include_after_loop_termination(tmp_path, control):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        f"import leaf\n\ndef run(value):\n    for _ in range(1):\n        {control}\n        leaf.run(value)\n    return value\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="unsupported For control flow"):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    "body",
    [
        "try:\n        return value\n    finally:\n        pass\n    return leaf.run(value)",
        "while True:\n        return value\n    return leaf.run(value)",
        "while True:\n        pass\n    return leaf.run(value)",
        "try:\n        pass\n    except Exception:\n        return leaf.run(value)\n    return value",
        (
            "try:\n        pass\n    finally:\n        if stop:\n            return value\n"
            "    if stop:\n        return leaf.run(value)\n    return value"
        ),
        (
            "while True:\n        try:\n            return value\n        except Exception:\n"
            "            break\n    return leaf.run(value)"
        ),
        "for _ in (1,):\n        break\n    else:\n        return leaf.run(value)\n    return value",
    ],
)
def test_wrapper_composition_rejects_unsupported_structured_control_flow(tmp_path, body):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition("op", f"import leaf\n\ndef run(value, stop=False):\n    {body}\n", ["leaf"])
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="unsupported .* control flow"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_rejects_loop_even_when_following_call_is_reachable(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value):\n    while True:\n        break\n    return leaf.run(value)\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="unsupported While control flow"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_rejects_deleted_boolean_predicate_input(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        (
            "import leaf\n\n"
            "def run(value, use_leaf):\n"
            "    del use_leaf\n"
            "    if use_leaf:\n"
            "        return leaf.run(value)\n"
            "    return value\n"
        ),
        ["leaf"],
    )
    wrapper["inputs"] = {"use_leaf": {"shape": None, "dtype": "bool"}}
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="unsupported|assigned local names"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_rejects_delete_before_include_argument_evaluation(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value):\n    del value\n    return leaf.run(value)\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="unsupported Delete control flow"):
        validate_definition_composition(wrapper, wrapper_path)


@pytest.mark.parametrize(
    "run_declaration",
    [
        "def run(value=leaf.run(1)):\n    return value\n",
        "def run(value: leaf.run(1)):\n    return value\n",
        "@leaf.run(1)\ndef run(value):\n    return value\n",
    ],
)
def test_wrapper_composition_rejects_include_calls_outside_run_body(tmp_path, run_declaration):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition("op", f"import leaf\n\n{run_declaration}", ["leaf"])
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match="only be referenced inside run body"):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_accepts_include_reachable_in_one_boolean_assignment(tmp_path):
    leaf = _definition("leaf", "def run(value):\n    return value\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value, use_leaf):\n    if use_leaf:\n        return leaf.run(value)\n    return value\n",
        ["leaf"],
    )
    wrapper["inputs"] = {
        "value": {"shape": None, "dtype": "float32"},
        "use_leaf": {"shape": None, "dtype": "bool"},
    }
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_binds_distinct_semantic_variants_of_one_raw_kernel(tmp_path):
    bf16_name = "raw_kernel__beta_bf16"
    fp32_name = "raw_kernel__beta_fp32"
    bf16_leaf = _definition(bf16_name, "def run(beta):\n    return beta\n")
    fp32_leaf = _definition(fp32_name, "def run(beta, post_sigmoid):\n    return beta if post_sigmoid else -beta\n")
    wrapper = _definition(
        "op",
        (
            f"import {bf16_name}\n"
            f"import {fp32_name}\n\n"
            "def run(beta, use_post_sigmoid):\n"
            "    if use_post_sigmoid:\n"
            f"        return {fp32_name}.run(beta, post_sigmoid=True)\n"
            f"    return {bf16_name}.run(beta)\n"
        ),
        [bf16_name, fp32_name],
    )
    wrapper["inputs"] = {
        "beta": {"shape": None, "dtype": "float32"},
        "use_post_sigmoid": {"shape": None, "dtype": "bool"},
    }
    wrapper_path = _write(_hierarchical_path(tmp_path, "wrapper"), wrapper)
    _write(wrapper_path.parent / f"{bf16_name}.json", bf16_leaf)
    _write(wrapper_path.parent / f"{fp32_name}.json", fp32_leaf)

    validate_definition_composition(wrapper, wrapper_path)
    with installed_reference_modules(wrapper, wrapper_path):
        namespace = {}
        exec(wrapper["reference"], namespace)
        assert namespace["run"](3, False) == 3
        assert namespace["run"](3, True) == 3


@pytest.mark.parametrize(
    ("leaf_signature", "call", "match"),
    [
        ("value", "leaf.run(*values)", "must not use \\*args expansion"),
        ("value, /", "leaf.run(value=value)", "positional-only"),
        ("value, scale", "leaf.run(value, value=scale, scale=scale)", "more than once"),
        ("value", "leaf.run(value, value)", "too many positional"),
        ("value", "leaf.run(value, unknown=1)", "unknown keyword"),
        ("value", "leaf.run(value=value, value=scale)", "duplicate keyword"),
        ("value, *, scale", "leaf.run(value)", "missing required"),
    ],
)
def test_wrapper_composition_enforces_full_leaf_call_binding(tmp_path, leaf_signature, call, match):
    leaf = _definition("leaf", f"def run({leaf_signature}):\n    return value\n")
    wrapper = _definition(
        "op",
        f"import leaf\n\ndef run(value, scale=1, values=()):\n    return {call}\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    with pytest.raises(DefinitionCompositionError, match=match):
        validate_definition_composition(wrapper, wrapper_path)


def test_wrapper_composition_matches_positional_only_keyword_capture_semantics(tmp_path):
    leaf = _definition("leaf", "def run(value, /, **kwargs):\n    return value, kwargs\n")
    wrapper = _definition(
        "op",
        "import leaf\n\ndef run(value, shadow):\n    return leaf.run(value, value=shadow)\n",
        ["leaf"],
    )
    wrapper_path = _write(tmp_path / "op.json", wrapper)
    _write(tmp_path / "leaf.json", leaf)
    validate_definition_composition(wrapper, wrapper_path)
