# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from tests.kernel_inventory.runtime_inputs import RuntimeInputCatalog
from tests.kernel_inventory.runtime_inputs import make_runtime_override
from tests.kernel_inventory.runtime_inputs import resolve_torch_dtype


def test_runtime_input_patterns_merge_without_tensor_metadata(tmp_path):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        """version: 1
patterns:
  dense:
    axes: {B: 1, T: 8}
    mutates: [state]
    inputs:
      scale: {kind: full, value: 0.5}
cases:
  src/tilegym/suites/example::definition::op/cutile/leaf:
    patterns: [dense]
    axes: {T: 16}
    mutates: [cache]
    inputs:
      offsets: {kind: arange, start: 0, step: 16}
""",
        encoding="utf-8",
    )
    catalog = RuntimeInputCatalog.from_path(path)
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"

    case = catalog.case_for_definition(definition_path)

    assert case.axes == {"B": 1, "T": 16}
    assert case.inputs["scale"] == {"kind": "full", "value": 0.5}
    assert "shape" not in case.inputs["offsets"]
    assert "dtype" not in case.inputs["offsets"]
    assert case.mutated_inputs == ("state", "cache")


def test_runtime_input_catalog_rejects_missing_patterns(tmp_path):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        """version: 1
patterns: {}
cases:
  src/tilegym/suites/example::definition::op/cutile/leaf:
    patterns: [missing]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing patterns"):
        RuntimeInputCatalog.from_path(path)


def test_runtime_input_catalog_rejects_duplicate_mutations(tmp_path):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        """version: 1
patterns:
  invalid:
    mutates: [state, state]
cases: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique input names"):
        RuntimeInputCatalog.from_path(path)


@pytest.mark.parametrize(
    ("catalog", "match"),
    [
        ("version: true\npatterns: {}\ncases: {}\n", "version must be 1"),
        ("version: 1\npatterns: {bad: {axes: {N: true}}}\ncases: {}\n", "positive integers"),
        (
            "version: 1\npatterns: {}\ncases:\n"
            "  src/tilegym/suites/example::definition::op/cutile/leaf:\n"
            "    axes: {N: true}\n",
            "positive integers",
        ),
    ],
)
def test_runtime_input_catalog_rejects_boolean_integer_fields(tmp_path, catalog, match):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(catalog, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        RuntimeInputCatalog.from_path(path)


def test_runtime_input_catalog_rejects_boolean_override_equal_to_const_axis_one(tmp_path):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns: {{}}
cases:
  {canonical_id}:
    axes: {{N: true}}
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps({"axes": {"N": {"type": "const", "value": 1}}, "inputs": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive integers"):
        RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])


@pytest.mark.parametrize("field", ["shape: [N]", "dtype: float64", "expression: arbitrary()"])
def test_runtime_input_catalog_rejects_generator_metadata_and_unknown_fields(tmp_path, field):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        f"""version: 1
patterns:
  invalid:
    inputs:
      value: {{kind: zeros, {field}}}
cases: {{}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generator has unsupported fields"):
        RuntimeInputCatalog.from_path(path)


@pytest.mark.parametrize(
    ("generator", "match"),
    [
        ("{kind: normal, mean: bad}", "normal mean/std"),
        ("{kind: normal, std: -1}", "normal mean/std"),
        ("{kind: arange, step: 0}", "arange start/step"),
        ("{kind: randint, low: 1.5, high: 4}", "randint low/high"),
        ("{kind: randint, low: 4, high: 4}", "randint low/high"),
    ],
)
def test_runtime_input_catalog_rejects_invalid_generator_parameters(tmp_path, generator, match):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        f"""version: 1
patterns:
  invalid:
    inputs:
      value: {generator}
cases: {{}}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        RuntimeInputCatalog.from_path(path)


@pytest.mark.parametrize(
    "duplicate",
    [
        "patterns:\n  shared: {}\n  shared: {}\ncases: {}",
        "patterns:\n  shared:\n    axes: {T: 4, T: 8}\ncases: {}",
        "patterns:\n  shared:\n    inputs: {scale: {kind: full, value: 1}, scale: {kind: full, value: 2}}\ncases: {}",
        (
            "patterns: {}\ncases:\n"
            "  src/tilegym/suites/example::definition::op/cutile/op: {}\n"
            "  src/tilegym/suites/example::definition::op/cutile/op: {}"
        ),
    ],
)
def test_runtime_input_catalog_rejects_duplicate_yaml_keys(tmp_path, duplicate):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(f"version: 1\n{duplicate}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        RuntimeInputCatalog.from_path(path)


def test_runtime_input_catalog_rejects_stale_canonical_definition_ids(tmp_path):
    path = tmp_path / "runtime_inputs.yaml"
    path.write_text(
        """version: 1
patterns: {}
cases:
  src/tilegym/suites/example::definition::op/cutile/stale: {}
""",
        encoding="utf-8",
    )
    existing = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale canonical Definition ids"):
        RuntimeInputCatalog.from_path(path, definition_paths=[existing])


def test_runtime_input_catalog_reuses_one_pattern_for_public_and_wrappers(tmp_path):
    path = tmp_path / "runtime_inputs.yaml"
    ids = [
        "src/tilegym/suites/example::definition::op/op",
        "src/tilegym/suites/example::definition::op/cutile/op",
        "src/tilegym/suites/example::definition::op/triton/op",
    ]
    path.write_text(
        """version: 1
patterns:
  wrapper_case:
    axes: {T: 8}
    inputs:
      enabled: {kind: full, value: true}
cases:
  src/tilegym/suites/example::definition::op/op:
    patterns: [wrapper_case]
  src/tilegym/suites/example::definition::op/cutile/op:
    patterns: [wrapper_case]
  src/tilegym/suites/example::definition::op/triton/op:
    patterns: [wrapper_case]
""",
        encoding="utf-8",
    )
    definition_paths = []
    for canonical_id in ids:
        relative = canonical_id.split("::definition::", 1)[1]
        definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions" / f"{relative}.json"
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(
            json.dumps(
                {
                    "axes": {"T": {"type": "var"}},
                    "inputs": {"enabled": {"shape": None, "dtype": "bool"}},
                }
            ),
            encoding="utf-8",
        )
        definition_paths.append(definition_path)

    catalog = RuntimeInputCatalog.from_path(path, definition_paths=definition_paths)
    cases = [catalog.case_for_definition(definition_path) for definition_path in definition_paths]
    assert [case.axes for case in cases] == [{"T": 8}] * 3
    assert [case.inputs["enabled"] for case in cases] == [{"kind": "full", "value": True}] * 3


@pytest.mark.parametrize(
    ("pattern_body", "match"),
    [
        ("axes: {Missing: 2}", "unknown axes"),
        ("inputs: {missing: {kind: zeros}}", "unknown inputs"),
        ("mutates: [missing]", "mutates unknown inputs"),
        ("axes: {D: 8}", "contradicts const axes"),
    ],
)
def test_runtime_input_catalog_statically_validates_merged_case_against_definition(tmp_path, pattern_body, match):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns:
  shared:
    {pattern_body}
cases:
  {canonical_id}:
    patterns: [shared]
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps(
            {
                "axes": {"N": {"type": "var"}, "D": {"type": "const", "value": 4}},
                "inputs": {"value": {"shape": ["N", "D"], "dtype": "float32"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])


@pytest.mark.parametrize("value", ["nope", "[true]", "2147483648"])
def test_runtime_input_catalog_rejects_schema_incompatible_scalar_values(tmp_path, value):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns: {{}}
cases:
  {canonical_id}:
    inputs:
      mode: {{kind: full, value: {value}}}
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps(
            {
                "axes": {},
                "inputs": {"mode": {"shape": None, "dtype": "int32"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incompatible with scalar Definition dtype int32"):
        RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])


def test_runtime_input_catalog_supports_explicit_optional_scalar_none_override(tmp_path):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns: {{}}
cases:
  {canonical_id}:
    inputs:
      optional: {{kind: none}}
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps(
            {
                "axes": {},
                "inputs": {"optional": {"shape": None, "dtype": "float32"}},
                "reference": "def run(optional=None):\n    return optional\n",
            }
        ),
        encoding="utf-8",
    )

    catalog = RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])

    assert catalog.case_for_definition(definition_path).inputs["optional"] == {"kind": "none"}
    assert (
        make_runtime_override(
            {"kind": "none"},
            {"shape": None, "dtype": "float32"},
            {},
            torch=None,
            device=None,
        )
        is None
    )


def test_runtime_input_catalog_supports_explicit_optional_tensor_none_override(tmp_path):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns: {{}}
cases:
  {canonical_id}:
    inputs:
      value: {{kind: none}}
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps(
            {
                "axes": {"N": {"type": "var"}},
                "inputs": {"value": {"shape": ["N"], "dtype": "float32"}},
                "reference": "def run(value=None):\n    return value\n",
            }
        ),
        encoding="utf-8",
    )
    catalog = RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])
    assert catalog.case_for_definition(definition_path).inputs["value"] == {"kind": "none"}


def test_runtime_input_catalog_rejects_none_override_without_reference_default(tmp_path):
    canonical_id = "src/tilegym/suites/example::definition::op/cutile/leaf"
    catalog_path = tmp_path / "runtime_inputs.yaml"
    catalog_path.write_text(
        f"""version: 1
patterns: {{}}
cases:
  {canonical_id}:
    inputs:
      required: {{kind: none}}
""",
        encoding="utf-8",
    )
    definition_path = tmp_path / "src/tilegym/suites/example/kernel_definitions/op/cutile/leaf.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(
        json.dumps(
            {
                "axes": {},
                "inputs": {"required": {"shape": None, "dtype": "float32"}},
                "reference": "def run(required):\n    return required\n",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit reference.run default of None"):
        RuntimeInputCatalog.from_path(catalog_path, definition_paths=[definition_path])


def test_runtime_override_derives_tensor_shape_and_dtype_from_definition():
    torch = pytest.importorskip("torch")
    value = make_runtime_override(
        {"kind": "arange", "start": 1, "step": 2},
        {"shape": ["B", "T"], "dtype": "int32"},
        {"B": 1, "T": 3},
        torch,
        torch.device("cpu"),
    )
    assert tuple(value.shape) == (1, 3)
    assert value.dtype == torch.int32
    assert value.tolist() == [[1, 3, 5]]


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_runtime_dtype_resolver_covers_validated_dtype_families(dtype):
    torch = pytest.importorskip("torch")

    assert resolve_torch_dtype(dtype, torch) is getattr(torch, dtype)


def test_runtime_override_materializes_unsigned_integer_generators():
    torch = pytest.importorskip("torch")
    axes = {"N": 3}
    device = torch.device("cpu")

    ranged = make_runtime_override(
        {"kind": "randint", "low": 1, "high": 4},
        {"shape": ["N"], "dtype": "uint32"},
        axes,
        torch,
        device,
    )
    sequenced = make_runtime_override(
        {"kind": "arange", "start": 1, "step": 2},
        {"shape": ["N"], "dtype": "uint64"},
        axes,
        torch,
        device,
    )

    assert ranged.dtype == torch.uint32
    assert all(1 <= value < 4 for value in ranged.tolist())
    assert sequenced.dtype == torch.uint64
    assert sequenced.tolist() == [1, 3, 5]


def test_runtime_override_accepts_shape_checked_literal_values():
    torch = pytest.importorskip("torch")
    spec = {"shape": ["N", "TWO"], "dtype": "int64"}
    axes = {"N": 2, "TWO": 2}

    value = make_runtime_override(
        {"kind": "values", "value": [[0, 0], [1, 0]]},
        spec,
        axes,
        torch,
        torch.device("cpu"),
    )

    assert value.dtype == torch.int64
    assert value.tolist() == [[0, 0], [1, 0]]
    with pytest.raises(ValueError, match="expected"):
        make_runtime_override(
            {"kind": "values", "value": [[0, 0]]},
            spec,
            axes,
            torch,
            torch.device("cpu"),
        )
