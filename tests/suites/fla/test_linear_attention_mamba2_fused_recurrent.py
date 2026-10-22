# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Tests for the self-contained port of Mamba-2 (SSD) fused-recurrent linear attention.

Mamba-2 is simple-GLA with a per-token, data-dependent scalar decay alpha_t:
    S_t = alpha_t S_{t-1} + k_t^T v_t,   o_t = (q_t * scale) S_t.
The op consumes the per-token log-decay g = log(alpha), shape [B, T, H].
Reference `naive_mamba2` is the torch-only recurrence (matching the math in
`fla.ops.simple_gla.naive` and `linear_attention_mamba2.py`).
"""

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_mamba2_fused_recurrent as fused_recurrent_mamba2

_BACKENDS = ["cutile"]


def naive_mamba2(q, k, v, alpha, scale=None, initial_state=None, output_final_state=False, reverse=False):
    """Mamba-2 reference. q, k, v: [B, T, H, K/K/V]; alpha: [B, T, H].

    Optional features mirror the public op so coverage tests can compare against
    a torch-only oracle:
      - ``initial_state``: [B, H, K, V] seed for the recurrent state S.
      - ``output_final_state``: also return the state S after the last step.
      - ``reverse``: run the recurrence from t=T-1 down to t=0.
    """
    orig_type = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v = q.float(), k.float(), v.float()
    B, T, H, K = q.shape
    Vd = v.shape[-1]
    if initial_state is None:
        S = torch.zeros(B, H, K, Vd, device=q.device, dtype=torch.float32)
    else:
        S = initial_state.float().clone()
    o = torch.empty(B, T, H, Vd, device=q.device, dtype=torch.float32)
    steps = range(T - 1, -1, -1) if reverse else range(T)
    for t in steps:
        S = alpha[:, t].float()[..., None, None] * S + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale)
    if output_final_state:
        return o.to(orig_type), S
    return o.to(orig_type)


def make_inputs(B, T, H, K, V, dtype, device="cuda"):
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    # alpha in (0.85, 1.0): healthy data-dependent decay range.
    alpha = 0.85 + 0.15 * torch.rand(B, T, H, dtype=torch.float32, device=device)
    g = alpha.log().to(torch.float32)
    return q, k, v, alpha, g


SHAPES = [
    # (B, T, H, K, V)
    (1, 64, 2, 32, 32),
    (2, 128, 2, 48, 48),
    (2, 128, 4, 64, 64),
    (1, 256, 2, 64, 32),
    (2, 256, 2, 32, 64),
]


def _assert_close_rms(test, ref, ratio, name=""):
    test32, ref32 = test.float(), ref.float()
    rms_err = (test32 - ref32).square().mean().sqrt()
    rms_ref = ref32.square().mean().sqrt()
    err_ratio = (rms_err / (rms_ref + 1e-8)).item()
    assert err_ratio < ratio, f"{name} RMS err ratio {err_ratio:.3e} exceeds {ratio:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionMamba2FusedRecurrent(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``fused_recurrent_mamba2`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        ratio = 2e-3 if dtype == torch.float32 else 3e-2

        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=scale)
        ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, B, T, H, K, V, backend):
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v))

        qt, kt, vt = make_leaves()
        ot, _ = fused_recurrent_mamba2(qt, kt, vt, g, scale=scale)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        qr, kr, vr = make_leaves()
        orr = naive_mamba2(qr, kr, vr, alpha, scale=scale)
        orr.backward(grad)

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 5e-2, name=name)

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_scale_none_default(self, B, T, H, K, V, backend):
        """Forward with ``scale=None`` so the op computes its own ``K**-0.5``."""
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=None)
        ref = naive_mamba2(q, k, v, alpha, scale=None)
        _assert_close_rms(o, ref, 2e-3, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_state(self, B, T, H, K, V, dtype, backend):
        """Forward seeded with a non-zero ``initial_state`` (USE_INITIAL_STATE)."""
        self._prep(backend)
        ratio = 2e-3 if dtype == torch.float32 else 3e-2
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=scale, initial_state=h0)
        ref = naive_mamba2(q, k, v, alpha, scale=scale, initial_state=h0)
        _assert_close_rms(o, ref, ratio, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_output_final_state(self, B, T, H, K, V, backend):
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda")
        o, ht = fused_recurrent_mamba2(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )
        ref_o, ref_ht = naive_mamba2(
            q,
            k,
            v,
            alpha,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
        )
        assert ht is not None and ht.shape == (B, H, K, V)
        _assert_close_rms(o, ref_o, 2e-3, name="o")
        _assert_close_rms(ht, ref_ht, 2e-3, name="final_state")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_reverse(self, B, T, H, K, V, backend):
        """Forward with ``reverse=True`` (REVERSE branch of the kernel)."""
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=scale, reverse=True)
        ref = naive_mamba2(q, k, v, alpha, scale=scale, reverse=True)
        _assert_close_rms(o, ref, 2e-3, name="o")

    @pytest.mark.parametrize("K, V", [(96, 96), (128, 64), (64, 128)])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_multi_block_kv(self, K, V, backend):
        """Forward with K or V > 64 so NK/NV > 1 (multi K/V-block reduction)."""
        self._prep(backend)
        B, T, H = 2, 96, 2
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=scale)
        ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o, ref, 2e-3, name="o")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_reverse(self, B, T, H, K, V, backend):
        """Backward through the ``reverse=True`` path (REVERSE in bwd kernel)."""
        self._prep(backend)
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, torch.float32)
        scale = K**-0.5

        def make_leaves():
            return tuple(x.clone().detach().requires_grad_(True) for x in (q, k, v))

        qt, kt, vt = make_leaves()
        ot, _ = fused_recurrent_mamba2(qt, kt, vt, g, scale=scale, reverse=True)
        grad = torch.randn_like(ot)
        ot.backward(grad)

        qr, kr, vr = make_leaves()
        orr = naive_mamba2(qr, kr, vr, alpha, scale=scale, reverse=True)
        orr.backward(grad)

        for name, gt, gr in (("dq", qt.grad, qr.grad), ("dk", kt.grad, kr.grad), ("dv", vt.grad, vr.grad)):
            _assert_close_rms(gt, gr, 5e-2, name=name)

    @pytest.mark.parametrize("seqlens", [[64, 64], [40, 88, 32], [128]])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_cu_seqlens_varlen(self, seqlens, backend):
        """Forward with ``cu_seqlens`` (IS_VARLEN packed variable-length path)."""
        self._prep(backend)
        H, K, V = 2, 32, 32
        total_T = sum(seqlens)
        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
            dtype=torch.int32,
            device="cuda",
        )
        q, k, v, alpha, g = make_inputs(1, total_T, H, K, V, torch.float32)
        scale = K**-0.5
        o, _ = fused_recurrent_mamba2(q, k, v, g, scale=scale, cu_seqlens=cu_seqlens)

        # Independent reference: scan each packed segment separately (fresh state).
        ref = torch.empty_like(o)
        start = 0
        for length in seqlens:
            end = start + length
            seg_o = naive_mamba2(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                alpha[:, start:end],
                scale=scale,
            )
            ref[:, start:end] = seg_o
            start = end
        _assert_close_rms(o, ref, 2e-3, name="o")

    PERF_SHAPES = [
        (1, 8192, 16, 32, 32),
        (2, 4096, 8, 32, 32),
        # (B, T, H, K, V) -- aligned with fla upstream benchmark SHAPE_CONFIGS (K = V = D).
        (1, 8192, 96, 128, 128),  # B1_T8192_H96_D128
        (2, 16384, 16, 128, 128),  # B2_T16384_H16_D128
        (4, 2048, 16, 128, 128),  # B4_T2048_H16_D128
        (4, 4096, 64, 128, 128),  # B4_T4096_H64_D128
        (8, 2048, 32, 256, 256),  # B8_T2048_H32_D256
        (8, 1024, 8, 64, 64),  # B8_T1024_H8_D64
    ]

    @pytest.mark.parametrize("B, T, H, K, V", PERF_SHAPES)
    @pytest.mark.parametrize("framework", _BACKENDS)
    def test_perf(self, B, T, H, K, V, framework, record_property):
        self.setUp()
        if not tilegym.is_backend_available(framework):
            pytest.skip(f"Backend {framework} is not available")
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

        dtype = torch.bfloat16
        q, k, v, alpha, g = make_inputs(B, T, H, K, V, dtype)
        scale = K**-0.5
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)

        framework_fn = lambda g=g, k=k, q=q, v=v: fused_recurrent_mamba2(q, k, v, g, scale=scale)[0]

        # Check numerical output before timing.
        with torch.no_grad():
            o_chk = framework_fn()
            o_ref = naive_mamba2(q, k, v, alpha, scale=scale)
        _assert_close_rms(o_chk, o_ref, 1e-1, name="o")
        del o_chk, o_ref

        # Reset gradients between timing samples and verify they are finite.
        _o = framework_fn()
        _o.backward(torch.randn_like(_o))
        for _n, _t in (("dq", q.grad), ("dk", k.grad), ("dv", v.grad)):
            assert torch.isfinite(_t).all(), (
                f"{framework} {_n} non-finite before perf; perf number would be meaningless"
            )
        for _leaf in (q, k, v):
            _leaf.grad = None
        del _o

        result = common.benchmark_framework(framework, framework_fn, grad_to_none=[q, k, v])
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "fused_recurrent_mamba2")

        del q, k, v, alpha, g, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
