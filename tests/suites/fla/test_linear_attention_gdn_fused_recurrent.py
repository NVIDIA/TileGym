# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test for fused_recurrent_gated_delta_rule (Gated DeltaNet / GDN) - self-contained
port from flash-linear-attention (fla).

The reference is a torch-only naive recurrence (no fla, no einops) ported from
`fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule`:
    S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = (q_t * scale) S_t
with alpha_t = exp(g_t). The naive ref does NOT L2-normalize q/k internally, and
its arg order is (q, k, v, beta, g). The kernel takes (q, k, v, g, beta) and with
use_qk_l2norm_in_kernel=True normalizes q/k inside; to compare we feed the ref
F.normalize(q), F.normalize(k).

The reference operates on [B, H, T, *] layout while the public op takes
[B, T, H, *]; the naive ref transposes internally.
"""

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_gdn_fused_recurrent as fused_recurrent_gated_delta_rule

_BACKENDS = ["cutile"]


# (B, T, H, K, V)
SHAPES = [
    (2, 128, 4, 64, 64),
    (1, 256, 2, 32, 48),
]


def naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None, initial_state=None, output_final_state=False):
    """Torch-only GDN reference. Inputs are [B, T, H, *], beta/g are [B, T, H]."""
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g))
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def _naive_gdn_vector_beta(q, k, v, beta, g, scale=None):
    """Torch-only GDN reference with a per-V-channel (vector) beta.

    Mirrors `naive_recurrent_gated_delta_rule` but `beta` is [B, T, H, V] and is
    applied element-wise on the V axis (no `[..., None]` broadcast), matching the
    kernel's IS_BETA_HEADWISE=False branch. Inputs are [B, T, H, *].
    """
    q, k, v, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, g))
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)  # [B, H, T, V]
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]  # [B, H, V]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta  # element-wise on V (vector beta)
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    o = o.transpose(1, 2).contiguous()
    return o, h


def rms_ratio(test, ref):
    test = test.float()
    ref = ref.float()
    denom = ref.pow(2).mean().sqrt()
    num = (test - ref).pow(2).mean().sqrt()
    return (num / (denom + 1e-12)).item()


def make_inputs(B, T, H, K, V, dtype):
    # STABLE inputs: q,k = randn; beta in (0,1); g = log(rand) (alpha in (0,1)).
    q = torch.randn(B, T, H, K, device="cuda")
    k = torch.randn(B, T, H, K, device="cuda")
    v = torch.randn(B, T, H, V, device="cuda")
    beta = torch.rand(B, T, H, device="cuda")
    g = torch.rand(B, T, H, device="cuda").clamp_min_(1e-6).log()
    return q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype), g.to(dtype)


class Test_FLA_LinearAttentionGdnFusedRecurrent(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable)."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_gva_default_beta(self, backend):
        self._prep(backend)
        torch.manual_seed(47)
        q = F.normalize(torch.randn(1, 17, 1, 32, device="cuda"), dim=-1)
        k = F.normalize(torch.randn_like(q), dim=-1)
        v = torch.randn(1, 17, 2, 32, device="cuda")
        g = -torch.rand(1, 17, 2, device="cuda")
        actual, state = fused_recurrent_gated_delta_rule(q, k, v, g=g, output_final_state=True)
        expected, expected_state = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=torch.ones_like(g), output_final_state=True
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state, expected_state)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_input_device_differs_from_current(self, backend):
        self._prep(backend)
        with torch.cuda.device(1):
            q, k, v, beta, g = make_inputs(1, 33, 1, 32, 32, torch.bfloat16)
            if backend == "cutile":
                q.requires_grad_()
            expected, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            if q.requires_grad:
                expected.sum().backward()
                expected_grad = q.grad.clone()
                q.grad = None
        with torch.cuda.device(0):
            actual, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            if q.requires_grad:
                actual.sum().backward()
                torch.testing.assert_close(q.grad, expected_grad)
            assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        tol = 1e-4 if dtype == torch.float32 else 1e-2
        scale = K**-0.5

        q, k, v, beta, g = make_inputs(B, T, H, K, V, dtype)

        # Reference: feed L2-normalized q,k since the kernel normalizes internally.
        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_op_cutile_dhu_tail_sentinel(self):
        """The default BF16 fused route must mask Dhu stores for a partial chunk.

        The cuTile fused-recurrent entry delegates this configuration to the
        chunk implementation. With T=33, Dhu's 64-row tile has 31 invalid tail
        rows; those rows must not scatter through the shaped ``dv2`` tensor's
        raw-flat view and overwrite a valid early gradient element.
        """
        self._prep("cutile")
        torch.manual_seed(23)
        B, T, H, K, V = 1, 33, 1, 32, 32
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V, torch.bfloat16)

        q_out = q.detach().clone().requires_grad_(True)
        k_out = k.detach().clone().requires_grad_(True)
        v_out = v.detach().clone().requires_grad_(True)
        beta_out = beta.detach().clone().requires_grad_(True)
        g_out = g.detach().clone().requires_grad_(True)
        o_out, _ = fused_recurrent_gated_delta_rule(
            q_out,
            k_out,
            v_out,
            g=g_out,
            beta=beta_out,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        q_ref = q.detach().float().clone().requires_grad_(True)
        k_ref = k.detach().float().clone().requires_grad_(True)
        v_ref = v.detach().float().clone().requires_grad_(True)
        beta_ref = beta.detach().float().clone().requires_grad_(True)
        g_ref = g.detach().float().clone().requires_grad_(True)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            F.normalize(q_ref, p=2, dim=-1),
            F.normalize(k_ref, p=2, dim=-1),
            v_ref,
            beta_ref,
            g_ref,
            scale=scale,
            output_final_state=False,
        )

        do = torch.randn_like(o_out)
        o_out.backward(do)
        o_ref.backward(do.float())

        ratio = rms_ratio(v_out.grad, v_ref.grad)
        assert ratio < 5e-2, f"dv RMS ratio {ratio:.3e} exceeds 5e-2"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_not_implemented(self, backend):
        """Unsupported training configurations fail at the public entry."""
        self._prep(backend)
        q, k, v, beta, g = make_inputs(1, 16, 2, 32, 32, torch.float32)
        q.requires_grad_(True)
        with pytest.raises(NotImplementedError, match="training"):
            fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta)

    # (B, T, H, K, V) -- perf shapes.
    PERF_SHAPES = [
        # (B, T, H, K, V) -- aligned with fla upstream benchmark SHAPE_CONFIGS (K = V = D).
        (1, 8192, 96, 128, 128),  # B1_T8192_H96_D128
        (2, 16384, 16, 128, 128),  # B2_T16384_H16_D128
        (4, 2048, 16, 128, 128),  # B4_T2048_H16_D128
        (4, 4096, 64, 128, 128),  # B4_T4096_H64_D128
        (8, 2048, 32, 256, 256),  # B8_T2048_H32_D256
        (8, 1024, 8, 64, 64),  # B8_T1024_H8_D64
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
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
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V, torch.bfloat16)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the active backend matches the naive reference on
        # this exact perf shape BEFORE timing it (mirrors test_op). A hard assert
        # here turns a silently-broken kernel (e.g. NaN output) into a loud
        # failure instead of a fast-but-garbage benchmark number.
        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref, k_ref, v.float(), beta.float(), g.float(), scale=scale, output_final_state=False
        )
        o_chk, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale, use_qk_l2norm_in_kernel=True)
        ratio = rms_ratio(o_chk, o_ref)
        assert ratio < 2e-2, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del q_ref, k_ref, o_ref, o_chk

        framework_fn = lambda beta=beta, g=g, k=k, q=q, v=v: fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )[0]

        result = common.benchmark_framework(framework, framework_fn)
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "fused_recurrent_gated_delta_rule")

        del q, k, v, beta, g, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    # -------------------------------------------------------------------------
    # Targeted coverage tests.
    #
    # The single pre-existing `test_op` only exercises one combination of kernel
    # heuristics: USE_G=True, headwise scalar beta (IS_BETA_HEADWISE=True),
    # USE_QK_L2NORM_IN_KERNEL=True, no initial/final state, STATE_V_FIRST=False,
    # H==HV (no GVA), and APPLY_BETA_SIGMOID=False. Every test below pins down a
    # distinct, previously-unexercised heuristic-driven branch of
    # `fused_recurrent_gated_delta_rule_fwd_kernel` and validates against an
    # independent torch recurrence (the in-file naive reference or a small
    # extension of it), not just shapes/finiteness.
    # -------------------------------------------------------------------------

    # (B, T, H, K, V) shapes reused for the coverage tests below.
    COV_SHAPES = [
        (2, 64, 4, 64, 64),
        (1, 128, 2, 32, 48),
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_initial_and_final_state(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        torch.manual_seed(11)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5

        q, k, v, beta, g = make_inputs(B, T, H, K, V, dtype)
        # h0 is consumed in fp32 by both kernel and reference.
        h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) * 0.1

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, h_ref = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
        )

        o_out, h_out = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            initial_state=h0.clone(),
            output_final_state=True,
        )

        assert h_out is not None
        assert h_out.shape == (B, H, K, V)
        assert rms_ratio(o_out, o_ref) < tol
        assert rms_ratio(h_out, h_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_state_v_first_final_state(self, B, T, H, K, V, dtype, backend):
        self._prep(backend)
        torch.manual_seed(12)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5

        q, k, v, beta, g = make_inputs(B, T, H, K, V, dtype)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, h_ref = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            output_final_state=True,
        )

        o_out, h_out = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True,
            state_v_first=True,
        )

        assert h_out is not None
        # state_v_first stores the state as [N, HV, V, K].
        assert h_out.shape == (B, H, V, K)
        assert rms_ratio(o_out, o_ref) < tol
        # Compare against the reference [K, V] state transposed to [V, K].
        assert rms_ratio(h_out, h_ref.transpose(-1, -2)) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_no_gate(self, B, T, H, K, V, dtype, backend):
        """g=None disables the decay branch (USE_G=False)."""
        self._prep(backend)
        torch.manual_seed(13)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5

        q, k, v, beta, _ = make_inputs(B, T, H, K, V, dtype)
        g_zero = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g_zero,
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=None,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_vector_beta(self, B, T, H, K, V, dtype, backend):
        """Per-channel vector beta selects the IS_BETA_HEADWISE=False path."""
        self._prep(backend)
        torch.manual_seed(14)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5

        q, k, v, _, g = make_inputs(B, T, H, K, V, dtype)
        beta_vec = torch.rand(B, T, H, V, device="cuda").to(dtype)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = _naive_gdn_vector_beta(
            q_ref,
            k_ref,
            v.float(),
            beta_vec.float(),
            g.float(),
            scale=scale,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta_vec,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("allow_neg_eigval", [False, True])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_beta_sigmoid_in_kernel(self, B, T, H, K, V, allow_neg_eigval, backend):
        self._prep(backend)
        torch.manual_seed(15)
        dtype = torch.float32
        tol = 1e-4
        scale = K**-0.5

        q, k, v, _, g = make_inputs(B, T, H, K, V, dtype)
        beta_logits = torch.randn(B, T, H, device="cuda", dtype=dtype)

        beta_ref = torch.sigmoid(beta_logits.float())
        if allow_neg_eigval:
            beta_ref = beta_ref * 2

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta_ref,
            g.float(),
            scale=scale,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta_logits,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("has_dt_bias", [False, True])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_gate_in_kernel(self, B, T, H, K, V, has_dt_bias, backend):
        """use_gate_in_kernel fuses -exp(A_log)*softplus(g[+dt_bias]) (USE_GATE_IN_KERNEL)."""
        self._prep(backend)
        torch.manual_seed(16)
        dtype = torch.float32
        tol = 1e-4
        scale = K**-0.5
        HV = H  # no GVA here.

        q, k, v, beta, g_raw = make_inputs(B, T, H, K, V, dtype)
        # Fresh raw pre-activation gate (not log-space) for the fused path.
        g_raw = torch.randn(B, T, H, device="cuda", dtype=dtype)
        A_log = torch.randn(HV, device="cuda", dtype=dtype)
        dt_bias = torch.randn(HV, device="cuda", dtype=dtype) if has_dt_bias else None

        # Host replica of -exp(A_log) * softplus(g + dt_bias).
        g_in = g_raw.float()
        if has_dt_bias:
            g_in = g_in + dt_bias.float().view(1, 1, HV)
        g_eff = -torch.exp(A_log.float()).view(1, 1, HV) * F.softplus(g_in)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g_eff,
            scale=scale,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g_raw,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize(
        "B, T, H, HV, K, V",
        [
            (2, 64, 2, 4, 64, 64),
            (1, 96, 1, 4, 32, 48),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_grouped_value_attention(self, B, T, H, HV, K, V, dtype, backend):
        """HV > H exercises Grouped Value Attention head mapping (i_h = i_hv // (HV//H))."""
        self._prep(backend)
        torch.manual_seed(17)
        tol = 1e-4 if dtype == torch.float32 else 2e-2
        scale = K**-0.5
        assert HV % H == 0
        rep = HV // H

        q = torch.randn(B, T, H, K, device="cuda")
        k = torch.randn(B, T, H, K, device="cuda")
        v = torch.randn(B, T, HV, V, device="cuda")
        beta = torch.rand(B, T, HV, device="cuda")
        g = torch.rand(B, T, HV, device="cuda").log()
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        beta, g = beta.to(dtype), g.to(dtype)

        # Expand q/k heads to HV by repeating each head `rep` times so each
        # value head sees the q/k head it shares (matching i_h = i_hv // rep).
        q_exp = q.float().repeat_interleave(rep, dim=2)
        k_exp = k.float().repeat_interleave(rep, dim=2)
        q_ref = F.normalize(q_exp, p=2, dim=-1)
        k_ref = F.normalize(k_exp, p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert o_out.shape == (B, T, HV, V)
        assert rms_ratio(o_out, o_ref) < tol

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    @pytest.mark.parametrize("B, T, H, K, V", COV_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_default_scale_no_l2norm(self, B, T, H, K, V, backend):
        """scale=None default + use_qk_l2norm_in_kernel=False."""
        self._prep(backend)
        torch.manual_seed(18)
        dtype = torch.float32
        tol = 1e-4
        scale = K**-0.5  # what the wrapper substitutes when scale=None.

        q, k, v, beta, g = make_inputs(B, T, H, K, V, dtype)

        # No L2 norm in kernel -> feed raw q/k to the reference.
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
        )

        o_out, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=None,
            use_qk_l2norm_in_kernel=False,
        )

        assert rms_ratio(o_out, o_ref) < tol
