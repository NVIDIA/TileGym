# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Test for chunk_gated_delta_rule (Gated DeltaNet / GDN) - self-contained port
from flash-linear-attention (fla).

The reference is a torch-only naive recurrence (no fla, no einops) ported from
`fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule`:
    S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = (q_t * scale) S_t
with alpha_t = exp(g_t). The naive ref does NOT L2-normalize q/k internally, and
its arg order is (q, k, v, beta, g). The chunk op takes (q, k, v, g, beta) and
with use_qk_l2norm_in_kernel=True normalizes q/k inside; to compare we feed the
ref F.normalize(q), F.normalize(k).

chunk_gated_delta_rule is bf16-only (it asserts against fp32 inputs in fla and
is numerically tuned for bf16), so inputs here are bf16; the naive ref runs in
fp32.
"""

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.fla import linear_attention_gdn_chunk as chunk_gated_delta_rule

_BACKENDS = ["cutile"]


# (B, T, H, K, V)
SHAPES = [
    (2, 65, 2, 4, 8),
    (1, 64, 1, 7, 16),
    (1, 128, 1, 8, 64),
    (1, 33, 1, 15, 8),
    (1, 64, 1, 16, 16),
    (1, 128, 1, 64, 8),
    (2, 128, 4, 64, 64),
    (1, 256, 2, 32, 48),
    (2, 128, 2, 64, 32),
    (1, 33, 1, 32, 32),  # partial chunk; catches masked Dhu tail stores
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


def rms_ratio(test, ref):
    test = test.float()
    ref = ref.float()
    denom = ref.pow(2).mean().sqrt()
    num = (test - ref).pow(2).mean().sqrt()
    return (num / (denom + 1e-12)).item()


def make_inputs(B, T, H, K, V):
    # STABLE bf16 inputs: q,k = randn; beta in (0,1); g = log(rand) (alpha in (0,1)).
    q = torch.randn(B, T, H, K, device="cuda")
    k = torch.randn(B, T, H, K, device="cuda")
    v = torch.randn(B, T, H, V, device="cuda")
    beta = torch.rand(B, T, H, device="cuda")
    g = torch.rand(B, T, H, device="cuda").clamp_min_(1e-6).log()
    dt = torch.bfloat16
    return q.to(dt), k.to(dt), v.to(dt), beta.to(dt), g.to(dt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class Test_FLA_LinearAttentionGdnChunk(common.PyTestCase):
    def _prep(self, backend):
        """setUp + select backend (skip if unavailable). The module-level
        ``chunk_gated_delta_rule`` dispatch op then routes to the active backend."""
        self.setUp()
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward(self, B, T, H, K, V, backend):
        self._prep(backend)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

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

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("B, T, H, K, V", SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward(self, B, T, H, K, V, backend):
        self._prep(backend)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

        # --- chunk op grads ---
        qk = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
        vk = v.detach().clone().requires_grad_(True)
        betak = beta.detach().clone().requires_grad_(True)
        gk = g.detach().clone().requires_grad_(True)
        o_out, _ = chunk_gated_delta_rule(
            qk,
            kk,
            vk,
            g=gk,
            beta=betak,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        do = torch.randn_like(o_out)
        o_out.backward(do)

        # --- reference grads (autograd through the naive recurrence) ---
        qr = q.detach().float().clone().requires_grad_(True)
        kr = k.detach().float().clone().requires_grad_(True)
        vr = v.detach().float().clone().requires_grad_(True)
        betar = beta.detach().float().clone().requires_grad_(True)
        gr = g.detach().float().clone().requires_grad_(True)
        q_ref = F.normalize(qr, p=2, dim=-1)
        k_ref = F.normalize(kr, p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            vr,
            betar,
            gr,
            scale=scale,
            output_final_state=False,
        )
        o_ref.backward(do.float())

        for name, got, ref in (
            ("dq", qk.grad, qr.grad),
            ("dk", kk.grad, kr.grad),
            ("dv", vk.grad, vr.grad),
            ("dg", gk.grad, gr.grad),
            ("dbeta", betak.grad, betar.grad),
        ):
            ratio = rms_ratio(got, ref)
            assert ratio < 5e-2, f"{name} RMS ratio {ratio:.3e} exceeds 5e-2"

    # ------------------------------------------------------------------
    # Coverage-gap tests (added to the existing class to follow file
    # conventions: direct kernel import, in-file naive reference, rms_ratio
    # helper, deterministic seeds, bf16 inputs / fp32 reference).
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("T", [33, 64])
    def test_op_cutile_dhu_tail_scatter(self, T):
        """Masked Dhu lanes must not overwrite a valid flat ``dv2`` element."""
        self._prep("cutile")
        from tilegym.suites.fla.cutile.linear_attention_gdn_chunk import chunk_gated_delta_rule_bwd_dhu

        B, H, K, V = 1, 1, 32, 32
        shape_k = (B, T, H, K)
        shape_v = (B, T, H, V)
        q = torch.zeros(shape_k, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        w = torch.zeros_like(q)
        do = torch.zeros(shape_v, device="cuda", dtype=torch.bfloat16)
        dv = torch.ones_like(do)

        _, _, dv2 = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            scale=K**-0.5,
        )

        torch.testing.assert_close(dv2, dv, atol=0.0, rtol=0.0)

    # K and/or V > 256 exercise the full-width ``[BK, BV]`` state tile in the chunk
    # state kernels (chunk_gated_delta_rule_fwd_kernel_h_blockdim64 /
    # bwd_kernel_dhu_blockdim64). A prior version hardcoded four 64-wide K sub-blocks
    # (b_h1..b_h4 / b_dh1..b_dh4), capping K <= 256 and SILENTLY dropping K[256:] in
    # BOTH fwd and bwd (K=320 -> ~0.2 fwd RMS). Verify fwd + every grad for K/V > 256.
    LARGE_KV_SHAPES = [
        (1, 256, 2, 320, 64),  # K=320 (NK=5) > 256
        (1, 256, 2, 256, 320),  # V=320 > 256
        (1, 256, 2, 320, 320),  # both > 256
    ]

    @pytest.mark.parametrize("B, T, H, K, V", LARGE_KV_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_large_k(self, B, T, H, K, V, backend):
        self._prep(backend)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

        qk = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
        vk = v.detach().clone().requires_grad_(True)
        betak = beta.detach().clone().requires_grad_(True)
        gk = g.detach().clone().requires_grad_(True)
        o_out, _ = chunk_gated_delta_rule(qk, kk, vk, g=gk, beta=betak, scale=scale, use_qk_l2norm_in_kernel=True)

        qr = q.detach().float().clone().requires_grad_(True)
        kr = k.detach().float().clone().requires_grad_(True)
        vr = v.detach().float().clone().requires_grad_(True)
        betar = beta.detach().float().clone().requires_grad_(True)
        gr = g.detach().float().clone().requires_grad_(True)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            F.normalize(qr, p=2, dim=-1),
            F.normalize(kr, p=2, dim=-1),
            vr,
            betar,
            gr,
            scale=scale,
            output_final_state=False,
        )
        assert rms_ratio(o_out, o_ref) < 1e-2, f"fwd K={K} V={V} RMS ratio {rms_ratio(o_out, o_ref):.3e} exceeds 1e-2"

        do = torch.randn_like(o_out)
        o_out.backward(do)
        o_ref.backward(do.float())
        for name, got, ref in (
            ("dq", qk.grad, qr.grad),
            ("dk", kk.grad, kr.grad),
            ("dv", vk.grad, vr.grad),
            ("dg", gk.grad, gr.grad),
            ("dbeta", betak.grad, betar.grad),
        ):
            ratio = rms_ratio(got, ref)
            assert ratio < 5e-2, f"{name} K={K} V={V} RMS ratio {ratio:.3e} exceeds 5e-2"

    # T deliberately NOT a multiple of the internal chunk size 64, so the last
    # chunk is partial and exercises boundary masking.
    PARTIAL_CHUNK_SHAPES = [
        (2, 96, 4, 64, 64),  # 96 = 64 + 32 -> one full + one half chunk
        (1, 200, 2, 32, 48),  # 200 = 3*64 + 8 -> tiny tail chunk
    ]

    @pytest.mark.parametrize("B, T, H, K, V", PARTIAL_CHUNK_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_partial_last_chunk(self, B, T, H, K, V, backend):
        """Forward when T is not a multiple of the chunk size (64)."""
        self._prep(backend)
        torch.manual_seed(0)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

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

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < 2e-2

    # (B, T, H, K, V) for the state-carry tests.
    STATE_SHAPES = [
        (2, 128, 4, 64, 64),
        (1, 192, 2, 32, 48),
    ]

    @pytest.mark.parametrize("B, T, H, K, V", STATE_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_initial_and_final_state(self, B, T, H, K, V, backend):
        """Forward with a non-zero `initial_state` and `output_final_state=True`."""
        self._prep(backend)
        torch.manual_seed(1)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)
        # initial_state is fp32 [N, HV, K, V]; here N=B, HV=H.
        h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, hT_ref = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
        )

        o_out, hT_out = chunk_gated_delta_rule(
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

        assert hT_out is not None, "output_final_state=True must return a state"
        assert rms_ratio(o_out, o_ref) < 2e-2
        assert rms_ratio(hT_out, hT_ref) < 2e-2

    # (B, T, H, HV, K, V) -- HV > H exercises grouped-value attention.
    GVA_SHAPES = [
        (2, 128, 2, 4, 64, 64),
        (1, 128, 2, 6, 32, 48),
    ]

    @pytest.mark.parametrize("B, T, H, HV, K, V", GVA_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_grouped_value_attention(self, B, T, H, HV, K, V, backend):
        """Forward with HV > H (grouped-value attention)."""
        self._prep(backend)
        torch.manual_seed(2)
        scale = K**-0.5
        group = HV // H
        q = torch.randn(B, T, H, K, device="cuda")
        k = torch.randn(B, T, H, K, device="cuda")
        v = torch.randn(B, T, HV, V, device="cuda")
        beta = torch.rand(B, T, HV, device="cuda")
        g = torch.rand(B, T, HV, device="cuda").log()
        dt = torch.bfloat16
        q, k, v, beta, g = q.to(dt), k.to(dt), v.to(dt), beta.to(dt), g.to(dt)

        # Reference is per-(value-)head: expand q/k heads to HV by repeating.
        q_exp = q.repeat_interleave(group, dim=2)
        k_exp = k.repeat_interleave(group, dim=2)
        q_ref = F.normalize(q_exp.float(), p=2, dim=-1)
        k_ref = F.normalize(k_exp.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

        assert o_out.shape == (B, T, HV, V)
        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("B, T, H, K, V", STATE_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_no_l2norm_in_kernel(self, B, T, H, K, V, backend):
        """Forward with `use_qk_l2norm_in_kernel=False`."""
        self._prep(backend)
        torch.manual_seed(3)
        scale = K**-0.5
        # Keep q/k modest so the un-normalized recurrence stays bf16-stable.
        q = torch.randn(B, T, H, K, device="cuda") * 0.5
        k = torch.randn(B, T, H, K, device="cuda") * 0.5
        v = torch.randn(B, T, H, V, device="cuda")
        beta = torch.rand(B, T, H, device="cuda")
        g = torch.rand(B, T, H, device="cuda").log()
        dt = torch.bfloat16
        q, k, v, beta, g = q.to(dt), k.to(dt), v.to(dt), beta.to(dt), g.to(dt)

        # No internal normalization -> reference gets raw (un-normalized) q/k.
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            beta.float(),
            g.float(),
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=False,
        )

        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("allow_neg_eigval", [False, True])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_beta_sigmoid_in_kernel(self, allow_neg_eigval, backend):
        """Forward with `use_beta_sigmoid_in_kernel=True` (and allow_neg_eigval)."""
        self._prep(backend)
        torch.manual_seed(4)
        B, T, H, K, V = 2, 128, 2, 64, 64
        scale = K**-0.5
        q, k, v, _, g = make_inputs(B, T, H, K, V)
        # beta_raw spans negatives/positives so the sigmoid is non-trivial.
        beta_raw = (torch.randn(B, T, H, device="cuda")).to(torch.bfloat16)

        sig_scale = 2.0 if allow_neg_eigval else 1.0
        beta_ref = sig_scale * torch.sigmoid(beta_raw.float())

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta_ref,
            g.float(),
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta_raw,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
        )

        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("H", [2, 8])
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_gate_in_kernel(self, backend, H):
        """Forward with `use_gate_in_kernel=True` (A_log / dt_bias path)."""
        self._prep(backend)
        torch.manual_seed(5)
        B, T, K, V = 2, 128, 64, 64
        scale = K**-0.5
        q, k, v, beta, _ = make_inputs(B, T, H, K, V)
        g_raw = torch.rand(B, T, H, device="cuda").to(torch.bfloat16)
        A_log = torch.randn(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(H, device="cuda", dtype=torch.float32)

        # Per-step log-decay the kernel forms before cumsum:
        #   g_step = -exp(A_log) * softplus(g_raw + dt_bias)
        sp = F.softplus(g_raw.float() + dt_bias)
        g_step = -torch.exp(A_log) * sp  # broadcast over [B, T, H]

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g_step,
            scale=scale,
            output_final_state=False,
        )

        o_out, _ = chunk_gated_delta_rule(
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

        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_forward_default_scale(self, backend):
        """Forward with `scale=None` (kernel defaults to 1/sqrt(K))."""
        self._prep(backend)
        torch.manual_seed(6)
        B, T, H, K, V = 2, 128, 4, 64, 64
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        # Reference with the expected default scale 1/sqrt(K).
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v.float(),
            beta.float(),
            g.float(),
            scale=K**-0.5,
            output_final_state=False,
        )

        o_out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=None,
            use_qk_l2norm_in_kernel=True,
        )

        assert rms_ratio(o_out, o_ref) < 2e-2

    @pytest.mark.parametrize("B, T, H, K, V", STATE_SHAPES)
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_op_backward_initial_state(self, B, T, H, K, V, backend):
        """Backward with a non-zero `initial_state` (dh0 path)."""
        self._prep(backend)
        torch.manual_seed(7)
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)
        h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32)

        qk = q.detach().clone().requires_grad_(True)
        kk = k.detach().clone().requires_grad_(True)
        vk = v.detach().clone().requires_grad_(True)
        betak = beta.detach().clone().requires_grad_(True)
        gk = g.detach().clone().requires_grad_(True)
        h0k = h0.detach().clone().requires_grad_(True)
        o_out, _ = chunk_gated_delta_rule(
            qk,
            kk,
            vk,
            g=gk,
            beta=betak,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            initial_state=h0k,
            output_final_state=False,
        )
        do = torch.randn_like(o_out)
        o_out.backward(do)

        qr = q.detach().float().clone().requires_grad_(True)
        kr = k.detach().float().clone().requires_grad_(True)
        vr = v.detach().float().clone().requires_grad_(True)
        betar = beta.detach().float().clone().requires_grad_(True)
        gr = g.detach().float().clone().requires_grad_(True)
        h0r = h0.detach().clone().requires_grad_(True)
        q_ref = F.normalize(qr, p=2, dim=-1)
        k_ref = F.normalize(kr, p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            vr,
            betar,
            gr,
            scale=scale,
            initial_state=h0r,
            output_final_state=False,
        )
        o_ref.backward(do.float())

        for name, got, ref in (
            ("dq", qk.grad, qr.grad),
            ("dk", kk.grad, kr.grad),
            ("dv", vk.grad, vr.grad),
            ("dg", gk.grad, gr.grad),
            ("dbeta", betak.grad, betar.grad),
            ("dh0", h0k.grad, h0r.grad),
        ):
            ratio = rms_ratio(got, ref)
            assert ratio < 5e-2, f"{name} RMS ratio {ratio:.3e} exceeds 5e-2"

    # (B, T, H, K, V) -- perf shapes (T multiple of the internal chunk size 64).
    PERF_SHAPES = [
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
        scale = K**-0.5
        q, k, v, beta, g = make_inputs(B, T, H, K, V)

        # Correctness gate: a perf measurement on a numerically wrong kernel is
        # meaningless. Verify the active backend matches the naive reference on
        # this exact perf shape BEFORE timing it (mirrors test_forward). A hard
        # assert here turns a silently-broken kernel (e.g. NaN output) into a
        # loud failure instead of a fast-but-garbage benchmark number.
        q_ref = F.normalize(q.float(), p=2, dim=-1)
        k_ref = F.normalize(k.float(), p=2, dim=-1)
        o_ref, _ = naive_recurrent_gated_delta_rule(
            q_ref, k_ref, v.float(), beta.float(), g.float(), scale=scale, output_final_state=False
        )
        o_chk, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, scale=scale, use_qk_l2norm_in_kernel=True)
        ratio = rms_ratio(o_chk, o_ref)
        assert ratio < 2e-2, (
            f"{framework} output mismatch before perf (rms_ratio={ratio:.3e}); perf number would be meaningless"
        )
        del q_ref, k_ref, o_ref, o_chk

        # Enable autograd on q/k/v/beta/g so that, under MODE=auto (default) or
        # MODE=backward, benchmark() also times the standalone backward kernel.
        # test_backward differentiates all five inputs (checks dq/dk/dv/dg/dbeta),
        # so all five carry grad here.
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        beta = beta.detach().requires_grad_(True)
        g = g.detach().requires_grad_(True)

        framework_fn = lambda beta=beta, g=g, k=k, q=q, v=v: chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )[0]

        # grad_to_none resets input grads between reps so backward timing is not
        # polluted by gradient accumulation. In MODE=auto this records both
        # 'forward' and 'backward'; MODE=forward preserves the prior fwd-only result.
        result = common.benchmark_framework(framework, framework_fn, grad_to_none=[q, k, v, beta, g])
        record_property("benchmark", result)
        record_property("shape", f"({B}, {T}, {H}, {K}, {V})")
        record_property("op", "chunk_gated_delta_rule")

        del q, k, v, beta, g, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
