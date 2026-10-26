# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU contracts for host entry points, with GPU primitives replaced by stubs."""

import functools
import inspect
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
FLA = ROOT / "src/tilegym/suites/fla"
_HOST_BACKENDS = ["cutile"]


@pytest.fixture(scope="module", autouse=True)
def _cpu_package_imports():
    """Isolate backend discovery while importing the real modules and decorators."""
    import sys
    import types

    before = {name: value for name, value in sys.modules.items() if name == "tilegym" or name.startswith("tilegym.")}
    packages = ["tilegym", "tilegym.suites", "tilegym.suites.fla", "tilegym.suites.fla.cutile"]
    for name in packages:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
            sys.modules[name] = package
    yield
    for name in list(sys.modules):
        if (name == "tilegym" or name.startswith("tilegym.")) and name not in before:
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _patch_module_globals(monkeypatch):
    global _module_patch
    _module_patch = monkeypatch


def _import_module(name):
    import importlib

    if ".cutile.linear_attention_" in name:
        pytest.importorskip("cuda.tile")
    return importlib.import_module(name)


def _module_function(path, name, env):
    """Import an actual module and mock its collaborators without changing code."""

    module_name = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
    module = _import_module(module_name)
    for key, value in env.items():
        _module_patch.setattr(module, key, value, raising=False)
    result = getattr(module, name)
    env[name] = result
    return result


@pytest.mark.parametrize("suffix", ["chunk", "fused_chunk"])
def test_op_large_normalization_product_falls_back(suffix, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    module = _import_module(f"tilegym.suites.fla.cutile.linear_attention_{suffix}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(module.ct, "launch", lambda *args, **kwargs: pytest.fail("Unexpected GPU launch"))
    with FakeTensorMode():
        q = torch.empty((1, 131072, 128, 128), device="cuda", dtype=torch.bfloat16)
        product = module._normalization_product_or_eager(q, q, q, 0.5, None, None, False)
        assert product.shape == q.shape
        assert product.dtype == q.dtype
        assert product.device == q.device


@pytest.mark.parametrize(
    "suffix,callee",
    [
        ("gdn_chunk", "chunk_gated_delta_rule"),
    ],
)
def test_op_positional_dispatch(suffix, callee, monkeypatch):
    import importlib

    name = f"linear_attention_{suffix}"
    backend = _import_module(f"tilegym.suites.fla.cutile.{name}")
    ops = importlib.import_module("tilegym.suites.fla.ops")
    monkeypatch.setattr(backend, callee, lambda **kwargs: kwargs)
    public = getattr(ops, name)
    positional = [p for p in inspect.signature(public).parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
    values = {p.name: object() for p in positional}
    assert public(*values.values(), backend="cutile") == public(**values, backend="cutile")


@pytest.mark.parametrize(
    "suffix,entry,primitive",
    [
        ("chunk", "chunk_linear_attn", "chunk_simple_gla"),
        ("fused_chunk", "fused_chunk_linear_attn", "fused_chunk_simple_gla"),
    ],
)
@pytest.mark.parametrize("backend", _HOST_BACKENDS)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("carry", [False, True])
def test_op_final_state_contract(suffix, entry, primitive, backend, normalize, output_final_state, carry):
    def recurrence(**kwargs):
        return kwargs["v"].clone(), torch.ones(1, 1, 2, 2) if kwargs["output_final_state"] else None

    def cumsum(x, reverse=False, cu_seqlens=None, output_dtype=None):
        assert cu_seqlens is None
        result = x.flip(1).cumsum(1).flip(1) if reverse else x.cumsum(1)
        return result.to(output_dtype or x.dtype)

    env = {"torch": torch, primitive: recurrence, "chunk_global_cumsum": cumsum}
    path = FLA / backend / f"linear_attention_{suffix}.py"
    class_path = FLA / "cutile/_normalization.py" if backend == "cutile" else path
    _module_function(class_path, "StatefulNormalizeFunction", env)
    for name in ("normalize_with_z_state", entry):
        _module_function(path, name, env)
    q, k, v = [torch.ones(1, 3, 1, 2, requires_grad=True) for _ in range(3)]
    initial = torch.ones(1, 1, 2, 2) if carry else None
    z_init = torch.ones(1, 1, 1, 2) if carry else None
    if carry and normalize:
        initial = (initial, z_init)
    output, state = env[entry](
        q, k, v, normalize=normalize, output_final_state=output_final_state, initial_state=initial
    )
    if not output_final_state:
        assert state is None
    elif normalize:
        assert isinstance(state, tuple) and all(isinstance(x, torch.Tensor) for x in state)
        torch.testing.assert_close(state[1], k[:, -1:] * 3 + (z_init if carry else 0))
    else:
        assert isinstance(state, torch.Tensor)
    if normalize:
        denominator = (q.detach() * (2**-0.5) * (k.detach().cumsum(1) + (z_init if carry else 0))).sum(
            -1, keepdim=True
        ) + 1e-10
        torch.testing.assert_close(output, v.detach() / denominator)
    output.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    if normalize:
        assert q.grad is not None and k.grad is not None


def _host_utils():
    from collections import deque

    env = {"torch": torch, "functools": functools, "inspect": inspect, "deque": deque, "F": torch.nn.functional}
    for name in ("input_guard", "_segmented_arange"):
        _module_function(FLA / "cutile/_host_utils.py", name, env)
    return env


@pytest.mark.parametrize("skip", [False, True, ["b"]])
def test_op_input_guard_contiguity(skip):
    guard = _host_utils()["input_guard"]

    @guard(no_guard_contiguous=skip)
    def call(a, b):
        return a, b

    x = torch.arange(6).reshape(2, 3).t()
    a, b = call(x, b=x)
    assert a.is_contiguous() == (skip is not True)
    assert b.is_contiguous() == (skip is False)
    assert list(inspect.signature(call).parameters) == ["a", "b"]
    torch.testing.assert_close(a, x)
    torch.testing.assert_close(b, x)


@pytest.mark.parametrize("backend", _HOST_BACKENDS)
def test_op_sequence_metadata_tracks_inplace_updates(backend):
    env = {"torch": torch, "F": torch.nn.functional}
    path = FLA / backend / "linear_attention_chunk.py"
    _module_function(FLA / "cutile/_host_utils.py", "_segmented_arange", env)
    for name in ("prepare_lens", "prepare_chunk_indices", "prepare_chunk_offsets"):
        _module_function(path, name, env)
    sequence = torch.tensor([0, 2, 5])
    torch.testing.assert_close(env["prepare_lens"](sequence), torch.tensor([2, 3]))
    torch.testing.assert_close(env["prepare_chunk_indices"](sequence, 2), torch.tensor([[0, 0], [1, 0], [1, 1]]))
    torch.testing.assert_close(env["prepare_chunk_offsets"](sequence, 2), torch.tensor([0, 1, 3]))
    sequence[1] = 4
    torch.testing.assert_close(env["prepare_lens"](sequence), torch.tensor([4, 1]))
    torch.testing.assert_close(env["prepare_chunk_indices"](sequence, 2), torch.tensor([[0, 0], [0, 1], [1, 0]]))
    torch.testing.assert_close(env["prepare_chunk_offsets"](sequence, 2), torch.tensor([0, 2, 3]))


def test_op_segmented_arange_empty_segments():
    segments, offsets = _host_utils()["_segmented_arange"](torch.tensor([0, 2, 0, 1]))
    torch.testing.assert_close(segments, torch.tensor([1, 1, 3]))
    torch.testing.assert_close(offsets, torch.tensor([0, 1, 0]))
    segments, offsets = _host_utils()["_segmented_arange"](torch.tensor([], dtype=torch.long))
    assert segments.numel() == offsets.numel() == 0


@pytest.mark.parametrize("set_device", [False, True])
def test_op_input_guard_device_context(set_device):
    from contextlib import contextmanager
    from types import SimpleNamespace

    entered = []

    @contextmanager
    def device(index):
        entered.append(index)
        yield

    class Tensor:
        device = SimpleNamespace(type="cuda", index=2)

        def contiguous(self):
            return self

    env = {
        "torch": SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(device=device)),
        "functools": functools,
        "inspect": inspect,
    }
    guard = _module_function(FLA / "cutile/_host_utils.py", "input_guard", env)
    tensor = Tensor()
    call = guard(lambda *, x: x, set_device=set_device)
    assert call(x=tensor) is tensor
    assert entered == ([2] if set_device else [])


@pytest.mark.parametrize("backend", _HOST_BACKENDS)
@pytest.mark.parametrize("boundaries", [[0, 0, 2], [0, 2, 2], [0, 1, 1, 3], [0, 0, 0]])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("with_initial", [False, True])
def test_op_empty_normalization_state_and_gradients(backend, boundaries, reverse, with_initial):
    def scan(x, reverse=False, cu_seqlens=None, output_dtype=None):
        result = torch.empty_like(x, dtype=output_dtype or x.dtype)
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            part = x[:, a:b]
            result[:, a:b] = part.flip(1).cumsum(1).flip(1) if reverse else part.cumsum(1)
        return result

    env = {"torch": torch, "chunk_global_cumsum": scan}
    path = FLA / "cutile/_normalization.py" if backend == "cutile" else FLA / backend / "linear_attention_chunk.py"
    cls = _module_function(path, "StatefulNormalizeFunction", env)
    q, k, o = [torch.full((1, boundaries[-1], 1, 1), 2.0, requires_grad=True) for _ in range(3)]
    count = len(boundaries) - 1
    initial = (
        torch.arange(7, 7 + 4 * count, 4, dtype=torch.float).reshape(count, 1, 1, 1).requires_grad_()
        if with_initial
        else None
    )
    args = (o, q, k, 0.5, initial, reverse, torch.tensor(boundaries))
    actual, state = cls.apply(*args, scan) if backend == "cutile" else cls.apply(*args)
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, o)]
    zr = initial.detach().clone().requires_grad_() if with_initial else None
    pieces, states = [], []
    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        part = kr[:, a:b]
        cumulative = part.flip(1).cumsum(1).flip(1) if reverse else part.cumsum(1)
        if zr is not None:
            cumulative = cumulative + zr[i : i + 1]
        pieces.append(vr[:, a:b] / ((qr[:, a:b] * 0.5 * cumulative).sum(-1, keepdim=True) + 1e-10))
        states.append(
            cumulative[:, :1]
            if reverse and b > a
            else cumulative[:, -1:]
            if b > a
            else zr[i : i + 1]
            if zr is not None
            else k.new_zeros((1, 1, 1, 1))
        )
    expected, expected_state = torch.cat(pieces, dim=1), torch.cat(states, dim=0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)
    (actual.sum() + state.square().sum()).backward()
    (expected.sum() + expected_state.square().sum()).backward()
    for left, right in zip((q, k, o), (qr, kr, vr)):
        torch.testing.assert_close(left.grad, right.grad)
    if initial is not None:
        torch.testing.assert_close(initial.grad, zr.grad)


def _gdn_recurrent_env(backend="cutile"):
    from types import SimpleNamespace

    seen = []

    def apply(*args):
        seen.append(args)
        return args[2] * args[6].unsqueeze(-1), None

    env = {"torch": torch, "FusedRecurrentFunction": SimpleNamespace(apply=apply)}
    fn = _module_function(
        FLA / backend / "linear_attention_gdn_fused_recurrent.py", "fused_recurrent_gated_delta_rule", env
    )
    return fn, seen


@pytest.mark.parametrize("backend", _HOST_BACKENDS)
def test_op_gva_default_beta_matches_value_heads(backend):
    fn, seen = _gdn_recurrent_env(backend)
    q = torch.ones(1, 3, 1, 4)
    v = torch.ones(1, 3, 2, 4)
    output, _ = fn(q, q, v)
    explicit, _ = fn(q, q, v, beta=torch.ones(1, 3, 2))
    assert seen[0][6].shape == (1, 3, 2)
    torch.testing.assert_close(output, explicit)


@pytest.mark.parametrize("entry", ["chunk", "recurrent"])
@pytest.mark.parametrize("backend", _HOST_BACKENDS)
def test_op_gdn_rejects_unknown_options(entry, backend):
    if entry == "recurrent":
        fn, _ = _gdn_recurrent_env(backend)
    else:
        fn = _module_function(
            FLA / backend / "linear_attention_gdn_chunk.py", "chunk_gated_delta_rule", {"torch": torch}
        )
    x = torch.ones(1, 2, 1, 4)
    with pytest.raises(TypeError, match="Unexpected keyword"):
        fn(x, x, x, g=torch.ones(1, 2, 1), beta=torch.ones(1, 2, 1), use_gatte_in_kernel=True)


@pytest.mark.parametrize("option", ["dtype", "state", "final", "norm", "vector_beta", "varlen"])
@pytest.mark.parametrize("backend", _HOST_BACKENDS)
def test_op_gdn_training_restrictions_are_checked_at_entry(option, backend):
    fn, seen = _gdn_recurrent_env(backend)
    q = torch.ones(1, 2, 1, 4, dtype=torch.float32 if option == "dtype" else torch.bfloat16)
    v = q.clone().requires_grad_()
    kwargs = {"g": torch.ones(1, 2, 1), "use_qk_l2norm_in_kernel": option != "norm"}
    if option == "state":
        kwargs["initial_state"] = torch.zeros(1, 1, 4, 4)
    if option == "final":
        kwargs["output_final_state"] = True
    if option == "vector_beta":
        kwargs["beta"] = torch.ones_like(v)
    if option == "varlen":
        kwargs["cu_seqlens"] = torch.tensor([0, 2])
    with pytest.raises(NotImplementedError, match="training"):
        fn(q, q, v, **kwargs)
    assert not seen


def test_op_device_cache_key_distinguishes_device_and_architecture():
    from types import SimpleNamespace

    current = [0]
    arch = {0: (8, 0), 1: (10, 0)}
    env = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(current_device=lambda: current[0], get_device_capability=lambda d: arch[d])
        )
    }
    key = _module_function(FLA / "cutile/_host_utils.py", "device_cache_key", env)
    first = key()
    current[0] = 1
    assert key() != first
    current[0] = 0
    arch[0] = (9, 0)
    assert key() != first


@pytest.mark.parametrize("K,V", [(8, 64), (64, 8), (1, 1), (16, 16), (128, 128), (256, 64)])
def test_op_padded_output_tile_candidates(K, V):
    import importlib

    tuning = importlib.import_module("tilegym.suites.fla.cutile._tuning")
    candidates = tuning.output_tile_configs(K, V)
    assert candidates and len(candidates) == len(set(candidates))
    assert candidates[0] == (min(max(16, 1 << (K - 1).bit_length()), 64), min(max(16, 1 << (V - 1).bit_length()), 64))
    for suffix in ("gdn_chunk", "deltanet_chunk"):
        mod = _import_module(f"tilegym.suites.fla.cutile.linear_attention_{suffix}")
        assert mod._fwd_o_tile_configs(K, V) == candidates


@pytest.mark.parametrize("K", [32, 64, 128])
def test_op_mamba_chunk_routes_reject_gate_gradients(K, monkeypatch):
    import importlib

    recurrent = _import_module("tilegym.suites.fla.cutile.linear_attention_mamba2_fused_recurrent")
    chunk = _import_module("tilegym.suites.fla.cutile.linear_attention_mamba2_fused_chunk")
    ops = importlib.import_module("tilegym.suites.fla.ops")
    monkeypatch.setattr(recurrent, "_fused_chunk_simple_gla", lambda **kw: pytest.fail("Unsupported gate gradient"))
    monkeypatch.setattr(chunk, "fused_chunk_simple_gla", lambda **kw: pytest.fail("Unsupported gate gradient"))
    x = torch.ones(1, 64, 1, K, dtype=torch.bfloat16)
    g = torch.ones(1, 64, 1, requires_grad=True)
    for entry in (ops.linear_attention_mamba2_fused_chunk, ops.linear_attention_mamba2_fused_recurrent):
        with pytest.raises(NotImplementedError, match="gradients for g"):
            entry(x, x, x, g, backend="cutile")


@pytest.mark.parametrize("K", [64, 128, 256])
def test_op_mamba_chunk_routes_allow_wide_key_gradients(K, monkeypatch):
    import importlib

    recurrent = _import_module("tilegym.suites.fla.cutile.linear_attention_mamba2_fused_recurrent")
    chunk = _import_module("tilegym.suites.fla.cutile.linear_attention_mamba2_fused_chunk")
    ops = importlib.import_module("tilegym.suites.fla.ops")
    sentinel = object()
    monkeypatch.setattr(recurrent, "_fused_chunk_simple_gla", lambda **kw: sentinel)
    monkeypatch.setattr(chunk, "fused_chunk_simple_gla", lambda **kw: sentinel)
    x = torch.ones(1, 64, 1, K, dtype=torch.bfloat16, requires_grad=True)
    g = torch.ones(1, 64, 1)
    for entry in (ops.linear_attention_mamba2_fused_chunk, ops.linear_attention_mamba2_fused_recurrent):
        assert entry(x, x, x, g, backend="cutile") is sentinel


@pytest.mark.parametrize("disabled", [False, True])
def test_op_shared_tuning_failure_retry_and_device_cache(disabled, monkeypatch):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    tuning = importlib.import_module("tilegym.suites.fla.cutile._tuning")
    current, attempts, launches = [0], [], []
    cache = {}

    class Kernel:
        def replace_hints(self, **kwargs):
            return (self, tuple(kwargs.items()))

    kernel = Kernel()
    configs = tuning.hint_configs([1, 2])

    def search(*args):
        attempts.append(current[0])
        if len(attempts) == 1:
            raise RuntimeError("invalid device context")
        return SimpleNamespace(best=SimpleNamespace(config=args[0][1]))

    ct = SimpleNamespace(compiler_timeout=lambda _: nullcontext(), launch=lambda *args: launches.append(args))
    monkeypatch.setattr(tuning, "_runtime", lambda: (ct, search))
    monkeypatch.setattr(tuning, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(tuning, "_cache", cache)
    monkeypatch.setattr(tuning, "device_cache_key", lambda: (current[0], (10, 0)))
    monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "0")
    monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "1" if disabled else "0")
    for device in (0, 0, 0, 1, 0):
        current[0] = device
        tuning.tune_launch(kernel, None, lambda c: (c.occupancy,), lambda c: (), configs, ("shape",))
    if disabled:
        assert not attempts and not cache
        assert all(call[2] is kernel for call in launches)
    else:
        assert attempts == [0, 0, 1]
        assert len(cache) == 2
        assert launches[0][2] is kernel
        assert launches[1][2] is not kernel
    assert len(launches) == 5


def test_op_state_tiles_use_grid_capacity_and_safe_device_fallback(monkeypatch):
    from types import SimpleNamespace

    module = _import_module("tilegym.suites.fla.cutile.linear_attention_chunk")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(multi_processor_count=148))
    assert module._state_tile_kv(128, 128, 32, 0) == (64, 64)
    assert module._state_tile_kv(128, 128, 96, 0) == (128, 128)
    assert module._state_tile_kv(64, 128, 16, 0) == (32, 64)
    assert module._state_tile_kv(256, 256, 256, 0) == (128, 128)

    def unavailable(device):
        raise RuntimeError("Device properties unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    assert module._state_tile_kv(128, 128, 32, 0) == (128, 128)


def test_op_tuning_cache_hit_skips_candidate_construction(monkeypatch):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    tuning = importlib.import_module("tilegym.suites.fla.cutile._tuning")
    constructed, searches = [], []

    class Kernel:
        def replace_hints(self, **kwargs):
            return object()

    def candidates(occupancies):
        constructed.append(tuple(occupancies))
        return tuning.hint_configs(occupancies, (4,))

    def search(configs, *args):
        searches.append(1)
        return SimpleNamespace(best=SimpleNamespace(config=configs[-1]))

    ct = SimpleNamespace(compiler_timeout=lambda _: nullcontext(), launch=lambda *args: None)
    monkeypatch.setattr(tuning, "_runtime", lambda: (ct, search))
    monkeypatch.setattr(tuning, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(tuning, "_cache", {})
    monkeypatch.setattr(tuning, "device_cache_key", lambda: (0, (10, 0)))
    monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", "0")
    kernel = Kernel()
    for choices in ((1, 2), (1, 2), (1, 4)):
        tuning.tune_launch(
            kernel,
            None,
            lambda c: (1,),
            lambda c: (),
            lambda: candidates(choices),
            ("shape",),
            config_key=("hints", choices, (4,)),
        )
    assert constructed == [(1, 2), (1, 4)]
    assert len(searches) == 2


def test_op_tuning_public_switch_overrides_a_warm_cache(monkeypatch):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    tuning = importlib.import_module("tilegym.suites.fla.cutile._tuning")
    launches, searches = [], []

    class Kernel:
        def replace_hints(self, **kwargs):
            return object()

    kernel = Kernel()
    configs = tuning.hint_configs([1], (4,))

    def search(candidates, *args):
        searches.append(1)
        return SimpleNamespace(best=SimpleNamespace(config=candidates[0]))

    ct = SimpleNamespace(compiler_timeout=lambda _: nullcontext(), launch=lambda *args: launches.append(args[2]))
    monkeypatch.setattr(tuning, "_runtime", lambda: (ct, search))
    monkeypatch.setattr(tuning, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(tuning, "_cache", {})
    monkeypatch.setattr(tuning, "device_cache_key", lambda: (0, (10, 0)))
    for flag in ("0", "1", "0"):
        monkeypatch.setenv("TILEGYM_DISABLE_AUTOTUNE", flag)
        tuning.tune_launch(kernel, None, lambda c: (1,), lambda c: (), configs, ("shape",))
    assert len(searches) == 1
    assert launches[0] is launches[2]
    assert launches[1] is kernel
    assert launches[0] is not kernel
