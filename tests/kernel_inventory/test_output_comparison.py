# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.kernel_inventory.kernel_runtime_utils import _load_custom_comparator
from tests.kernel_inventory.kernel_runtime_utils import _load_reference_namespace
from tests.kernel_inventory.kernel_runtime_utils import _run_runtime_branch
from tilegym.kernel_inventory import _correctness_compat as compat
from tilegym.kernel_inventory import _workload_schema_compat as workload_compat

torch = pytest.importorskip("torch")

_COMPARATORS = """
def check_all(ref_outputs, user_outputs, *, atol=0.0):
    return {
        name: {"passed": bool(((user_outputs[name] - ref) .abs() <= atol).all())}
        for name, ref in ref_outputs.items()
    }


def check_only_values(ref_outputs, user_outputs):
    return {"values": {"passed": True}}


def check_raises(ref_outputs, user_outputs):
    raise RuntimeError("comparator exploded")


def check_mutates(ref_outputs, user_outputs):
    for tensor in (*ref_outputs.values(), *user_outputs.values()):
        tensor.fill_(0)
    return {name: {"passed": True} for name in ref_outputs}
"""


def _definition(entrypoint: str | None) -> dict[str, Any]:
    definition = {
        "name": "pair_op",
        "axes": {"N": {"type": "const", "value": 4}},
        "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {
            "values": {"shape": ["N"], "dtype": "float32"},
            "indices": {"shape": ["N"], "dtype": "int64"},
        },
        "reference": (
            "import torch\n\n"
            "def run(x):\n"
            "    values = x * 2\n"
            "    indices = torch.arange(x.shape[0], device=x.device)\n"
            "    return values, indices\n" + _COMPARATORS
        ),
    }
    if entrypoint is not None:
        definition["custom_correctness_entrypoint"] = entrypoint
    return definition


def _solution(offset: float = 0.0):
    def solution(x):
        values = x * 2 + offset
        indices = torch.arange(x.shape[0], device=x.device)
        return values, indices

    return solution


def _run(entrypoint: str | None, *, offset: float = 0.0, kwargs: dict[str, Any] | None = None) -> None:
    definition = _definition(entrypoint)
    namespace = _load_reference_namespace(definition, Path("pair_op.json"))
    _run_runtime_branch(
        definition,
        namespace["run"],
        {"spec": {"entry_point": "synthetic.py::solution"}},
        _solution(offset),
        {"x": torch.arange(4, dtype=torch.float32)},
        {"N": 4},
        torch,
        (),
        case_label="row",
        custom_comparator=_load_custom_comparator(definition, namespace),
        custom_correctness_kwargs=kwargs,
    )


def test_custom_comparator_pass():
    _run("check_all")


def test_custom_comparator_failure_is_reported():
    with pytest.raises(AssertionError, match="outputs rejected: values"):
        _run("check_all", offset=0.5)


def test_custom_comparator_kwargs_are_forwarded():
    _run("check_all", offset=0.5, kwargs={"atol": 1.0})


def test_custom_comparator_missing_output_fails():
    with pytest.raises(pytest.fail.Exception, match=r"missing=\['indices'\]"):
        _run("check_only_values")


def test_custom_comparator_exception_fails():
    with pytest.raises(pytest.fail.Exception, match="comparator exploded"):
        _run("check_raises")


def test_custom_comparator_mutating_outputs_does_not_change_verdict():
    # check_mutates zeroes its tensors; the following mutated-input and
    # unexpected-mutation checks must still see the original values.
    _run("check_mutates", offset=0.0)


def test_missing_custom_comparator_function_fails():
    with pytest.raises(pytest.fail.Exception, match="does not define it"):
        _run("check_absent")


def test_default_comparison_rejects_custom_kwargs():
    with pytest.raises(pytest.fail.Exception, match="custom_correctness_kwargs requires"):
        _run(None, kwargs={"atol": 1.0})


def test_output_shape_and_dtype_are_checked_before_custom_comparison():
    definition = _definition("check_all")
    namespace = _load_reference_namespace(definition, Path("pair_op.json"))

    def wrong_dtype(x):
        return x * 2, torch.arange(x.shape[0], device=x.device, dtype=torch.int32)

    with pytest.raises(AssertionError, match="has dtype"):
        _run_runtime_branch(
            definition,
            namespace["run"],
            {"spec": {"entry_point": "synthetic.py::wrong_dtype"}},
            wrong_dtype,
            {"x": torch.arange(4, dtype=torch.float32)},
            {"N": 4},
            torch,
            (),
            custom_comparator=_load_custom_comparator(definition, namespace),
        )


def _run_benchmark_only(solution_fn: Any, entrypoint: str | None) -> None:
    definition = _definition(entrypoint)
    namespace = _load_reference_namespace(definition, Path("pair_op.json"))
    _run_runtime_branch(
        definition,
        namespace["run"],
        {"spec": {"entry_point": "synthetic.py::solution"}},
        solution_fn,
        {"x": torch.arange(4, dtype=torch.float32)},
        {"N": 4},
        torch,
        (),
        case_label="row",
        skip_correctness=True,
        custom_comparator=_load_custom_comparator(definition, namespace),
    )


@pytest.mark.parametrize("entrypoint", [None, "check_raises"])
def test_benchmark_only_skips_numerical_comparison(entrypoint):
    _run_benchmark_only(_solution(offset=0.5), entrypoint)


def _benchmark_wrong_shape(x):
    return x[:2] * 2, torch.arange(x.shape[0], device=x.device)


def _benchmark_wrong_dtype(x):
    return x * 2, torch.arange(x.shape[0], device=x.device, dtype=torch.int32)


def _benchmark_wrong_structure(x):
    return x * 2


def _benchmark_mutates_input(x):
    values = x * 2
    x.add_(1)
    return values, torch.arange(x.shape[0], device=x.device)


@pytest.mark.parametrize("entrypoint", [None, "check_all"])
@pytest.mark.parametrize(
    ("solution_fn", "match"),
    [
        pytest.param(_benchmark_wrong_shape, "has shape", id="shape"),
        pytest.param(_benchmark_wrong_dtype, "has dtype", id="dtype"),
        pytest.param(_benchmark_wrong_structure, "must both return a tuple", id="structure"),
        pytest.param(_benchmark_mutates_input, "Solution unexpectedly mutated input x", id="input-mutation"),
    ],
)
def test_benchmark_only_still_enforces_output_and_input_contract(solution_fn, match, entrypoint):
    with pytest.raises(AssertionError, match=match):
        _run_benchmark_only(solution_fn, entrypoint)


# Synthetic default-comparison cases: (id, tolerance overrides, reference tensors, user tensors).
def _default_cases() -> list[Any]:
    ones = torch.ones(50)
    one_off = ones.clone()
    one_off[0] = 3.0
    two_off = one_off.clone()
    two_off[1] = 3.0
    ids = torch.arange(50, dtype=torch.int64)
    ids_off = ids.clone()
    ids_off[0] += 100
    neg_inf = torch.tensor([float("-inf"), 1.0])
    return [
        pytest.param({}, [ones], [ones.clone()], id="exact"),
        pytest.param({}, [ones], [ones + 0.015], id="within-atol"),
        pytest.param({}, [ones], [ones + 0.1], id="outside-tolerance"),
        pytest.param({"required_matched_ratio": 0.98}, [ones], [one_off], id="float-ratio-at-threshold"),
        pytest.param({"required_matched_ratio": 0.98}, [ones], [two_off], id="float-ratio-below-threshold"),
        pytest.param({"required_matched_ratio": 0.98}, [ids], [ids_off], id="int-ratio-at-threshold"),
        pytest.param({}, [ids], [ids_off], id="int-strict"),
        pytest.param({}, [ones], [torch.full((50,), float("nan"))], id="nan"),
        pytest.param({}, [torch.full((2,), float("inf"))], [torch.full((2,), float("inf"))], id="matching-inf"),
        pytest.param({}, [neg_inf], [neg_inf.clone()], id="matching-neg-inf"),
        pytest.param({"allow_negative_inf": True}, [neg_inf], [neg_inf.clone()], id="allowed-neg-inf"),
        pytest.param({}, [torch.full((4,), 1e-3)], [torch.zeros(4)], id="all-zero-solution"),
        pytest.param({"required_matched_ratio": 0.98, "max_error_cap": 1.0}, [ones], [one_off], id="max-error-cap"),
        pytest.param({}, [torch.empty(0)], [torch.empty(0)], id="empty"),
        pytest.param({}, [torch.tensor([True, False])], [torch.tensor([True, True])], id="bool-mismatch"),
        pytest.param({}, [ones.reshape(5, 10).t()], [ones.reshape(5, 10).t().clone()], id="non-contiguous"),
        pytest.param({}, [ones, ids], [ones.clone(), ids_off], id="multi-output"),
    ]


def _tolerance(module: Any, overrides: dict[str, Any]) -> Any:
    values = {"max_atol": 0.02, "max_rtol": 0.02, "required_matched_ratio": 1.0, **overrides}
    return module.ToleranceSpec(**values)


def _verdict(compare: Any, tolerance: Any, ref: list[Any], user: list[Any], **kwargs: Any) -> Any:
    names = [f"out{index}" for index in range(len(ref))]
    try:
        results = compare(ref, user, tolerance, names, **kwargs)
    except Exception as exc:
        return ("error", type(exc).__name__)
    return [(result.output_name, result.passed, result.has_nan, result.has_inf, result.metadata) for result in results]


_EXPECTED_DEFAULT_PASS = {
    "exact": True,
    "within-atol": True,
    "outside-tolerance": False,
    "float-ratio-at-threshold": True,
    "float-ratio-below-threshold": False,
    "int-ratio-at-threshold": True,
    "int-strict": False,
    "nan": False,
    "matching-inf": False,
    "matching-neg-inf": False,
    "allowed-neg-inf": True,
    "all-zero-solution": False,
    "max-error-cap": False,
    "empty": True,
    "bool-mismatch": False,
    "non-contiguous": True,
    "multi-output": False,
}


@pytest.mark.parametrize(("overrides", "ref", "user"), _default_cases())
def test_compat_default_comparison_semantics(request, overrides, ref, user):
    verdict = _verdict(compat.compare_outputs, _tolerance(workload_compat, overrides), ref, user)
    case = request.node.callspec.id

    assert all(passed for _, passed, *_ in verdict) is _EXPECTED_DEFAULT_PASS[case]


def _custom_cases() -> list[Any]:
    def passing(ref, user, **kwargs):
        return {name: {"passed": True} for name in ref}

    def failing(ref, user, **kwargs):
        return {name: {"passed": name != "out1", "max_absolute_error": 1.5} for name in ref}

    def missing(ref, user, **kwargs):
        return {"out0": {"passed": True}}

    def extra(ref, user, **kwargs):
        return {**{name: {"passed": True} for name in ref}, "bogus": {"passed": True}}

    def not_a_dict(ref, user, **kwargs):
        return [True, True]

    def bad_value(ref, user, **kwargs):
        return {name: True for name in ref}

    def renamed(ref, user, **kwargs):
        return {name: {"passed": True, "output_name": "other"} for name in ref}

    def bad_metadata(ref, user, **kwargs):
        return {name: {"passed": True, "max_absolute_error": -1.0} for name in ref}

    def raises(ref, user, **kwargs):
        raise RuntimeError("boom")

    def uses_kwargs(ref, user, *, threshold, nested):
        assert nested == {"values": [1, 2]}
        nested["values"].append(3)
        return {name: {"passed": float((user[name] - ref[name]).abs().max()) <= threshold} for name in ref}

    def mutates(ref, user, **kwargs):
        for tensor in (*ref.values(), *user.values()):
            tensor.fill_(7)
        return {name: {"passed": True} for name in ref}

    return [
        pytest.param(passing, None, id="pass"),
        pytest.param(failing, None, id="fail"),
        pytest.param(missing, None, id="missing-output"),
        pytest.param(extra, None, id="extra-output"),
        pytest.param(not_a_dict, None, id="non-dict"),
        pytest.param(bad_value, None, id="non-dict-value"),
        pytest.param(renamed, None, id="mismatched-output-name"),
        pytest.param(bad_metadata, None, id="invalid-metadata"),
        pytest.param(raises, None, id="raises"),
        pytest.param(uses_kwargs, {"threshold": 0.5, "nested": {"values": [1, 2]}}, id="kwargs"),
        pytest.param(mutates, None, id="mutates-inputs"),
    ]


def _custom_verdict(compare: Any, tolerance: Any, function: Any, kwargs: Any) -> Any:
    ref = [torch.ones(4), torch.arange(4)]
    user = [torch.ones(4) + 0.25, torch.arange(4)]
    snapshot = [tensor.clone() for tensor in (*ref, *user)]
    verdict = _verdict(
        compare, tolerance, ref, user, correctness_fn=function, entrypoint_name=None, custom_correctness_kwargs=kwargs
    )
    assert all(torch.equal(a, b) for a, b in zip((*ref, *user), snapshot, strict=True))
    if kwargs is not None and "nested" in kwargs:
        assert kwargs["nested"] == {"values": [1, 2]}
    return verdict


_EXPECTED_CUSTOM = {
    "pass": True,
    "fail": False,
    "missing-output": "error",
    "extra-output": "error",
    "non-dict": "error",
    "non-dict-value": "error",
    "mismatched-output-name": "error",
    "invalid-metadata": "error",
    "raises": "error",
    "kwargs": True,
    "mutates-inputs": True,
}


@pytest.mark.parametrize(("function", "kwargs"), _custom_cases())
def test_compat_custom_comparison_semantics(request, function, kwargs):
    verdict = _custom_verdict(compat.compare_outputs, _tolerance(workload_compat, {}), function, kwargs)
    expected = _EXPECTED_CUSTOM[request.node.callspec.id]

    if expected == "error":
        assert verdict == ("error", "ValueError")
    else:
        assert all(passed for _, passed, *_ in verdict) is expected
