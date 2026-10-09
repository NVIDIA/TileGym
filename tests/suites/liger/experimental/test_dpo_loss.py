# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import dpo_loss


def _sequence_logps(input_data, weight, target, bias=None, ignore_index=-100, average_log_prob=False):
    """Per-sequence target log-probs plus the tensors DPO derives from them.

    Returns (log_prob, log_probs, logits) where log_prob is (B,), log_probs is
    the float32 log-softmax over the vocab (B, T, V), and logits is (B, T, V)
    in the input dtype.
    """
    logits = input_data @ weight.t()
    if bias is not None:
        logits = logits + bias
    log_probs = F.log_softmax(logits.float(), dim=-1)

    loss_mask = target != ignore_index
    label = torch.where(loss_mask, target, 0)
    per_token_logps = log_probs.gather(-1, label.unsqueeze(-1)).squeeze(-1)
    if average_log_prob:
        log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        log_prob = (per_token_logps * loss_mask).sum(-1)
    return log_prob, log_probs, logits


def _preference_loss(
    chosen_logps,
    rejected_logps,
    n_pairs,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """Sigmoid DPO loss, transcribed from Liger-Kernel chunked_loss/dpo_loss.py
    (commit ead96b618e5c)."""
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios

    logits_diff = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits_diff).sum() / n_pairs
    return loss, chosen_rewards, rejected_rewards


def _reference_dpo_loss(
    input_data,
    weight,
    target,
    bias=None,
    ref_input=None,
    ref_weight=None,
    ref_bias=None,
    ignore_index=-100,
    beta=0.1,
    alpha=1.0,
    compute_nll_loss=False,
    use_ref_model=True,
    average_log_prob=False,
):
    """Unchunked PyTorch reference for the fused linear + DPO loss.

    Materializes the full (B, T, V) logits. Semantics follow Liger-Kernel's
    LigerFusedLinearDPOLoss (fused_linear_preference.py + dpo_loss.py at
    commit ead96b618e5c): the batch stacks chosen sequences then rejected
    sequences, per-sequence log-probs are summed (or averaged) over positions
    where target != ignore_index, and the preference loss is normalized by the
    number of pairs.
    """
    B, T, _H = input_data.shape
    V = weight.shape[0]
    n_pairs = B // 2

    log_prob, log_probs, logits = _sequence_logps(input_data, weight, target, bias, ignore_index, average_log_prob)
    chosen_logps = log_prob[:n_pairs]
    rejected_logps = log_prob[n_pairs:]

    nll_loss = torch.zeros((), device=input_data.device)
    if compute_nll_loss:
        nll_loss = F.nll_loss(
            log_probs[:n_pairs].reshape(-1, V),
            target[:n_pairs].reshape(-1),
            reduction="sum",
            ignore_index=ignore_index,
        )
        nll_loss = nll_loss / (target[:n_pairs] != ignore_index).sum()

    chosen_logits_mean = logits[:n_pairs].sum() / (n_pairs * T * V)
    rejected_logits_mean = logits[n_pairs:].sum() / (n_pairs * T * V)

    if use_ref_model:
        with torch.no_grad():
            ref_log_prob, _, _ = _sequence_logps(
                ref_input, ref_weight, target, ref_bias, ignore_index, average_log_prob
            )
        ref_chosen_logps = ref_log_prob[:n_pairs]
        ref_rejected_logps = ref_log_prob[n_pairs:]
    else:
        ref_chosen_logps = torch.zeros_like(chosen_logps)
        ref_rejected_logps = torch.zeros_like(rejected_logps)

    pref_loss, chosen_rewards, rejected_rewards = _preference_loss(
        chosen_logps,
        rejected_logps,
        n_pairs,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )

    loss = alpha * nll_loss + pref_loss
    return (
        loss,
        chosen_logps,
        rejected_logps,
        chosen_logits_mean,
        rejected_logits_mean,
        nll_loss,
        chosen_rewards,
        rejected_rewards,
    )


def _make_inputs(n_pairs, T, H, V, dtype, device, with_bias=False, with_ref=True, prompt_len=0, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    B = 2 * n_pairs
    input_data = randn(B, T, H) * 0.5
    weight = randn(V, H) * 0.1
    target = torch.randint(0, V, (B, T), generator=gen, dtype=torch.int64).to(device)
    if prompt_len > 0:
        target[:, :prompt_len] = -100
    bias = randn(V) * 0.1 if with_bias else None
    ref_input = randn(B, T, H) * 0.5 if with_ref else None
    ref_weight = randn(V, H) * 0.1 if with_ref else None
    ref_bias = randn(V) * 0.1 if (with_ref and with_bias) else None
    return input_data, weight, target, bias, ref_input, ref_weight, ref_bias


def _tols(dtype):
    tol = 5e-3 if dtype == torch.float32 else 1e-2
    return {"atol": tol, "rtol": tol}


def _assert_outputs_match(test_out, ref_out, dtype, what=""):
    names = (
        "loss",
        "chosen_logps",
        "rejected_logps",
        "chosen_logits_mean",
        "rejected_logits_mean",
        "nll_loss",
        "chosen_rewards",
        "rejected_rewards",
    )
    for name, t, r in zip(names, test_out, ref_out):
        assert torch.allclose(t.float(), r.float(), **_tols(dtype)), (
            f"{what}{name} mismatch: max={(t.float() - r.float()).abs().max().item():.6f}"
        )


class Test_Liger_DpoLoss(common.PyTestCase):
    _backends = ["cutile"]

    def _setup_backend(self, backend):
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.parametrize(
        "n_pairs, T, H, V, dtype",
        [
            (2, 8, 64, 128, torch.float32),
            (4, 16, 128, 256, torch.float32),
            (2, 8, 64, 128, torch.bfloat16),
            (2, 8, 64, 128, torch.float16),
            (3, 7, 41, 150, torch.float32),  # non-power-of-2 shapes
            # Shape from Liger test/chunked_loss/test_dpo_loss.py scale
            (4, 47, 31, 123, torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_forward(self, n_pairs, T, H, V, dtype, backend, monkeypatch):
        """Forward outputs match the PyTorch reference (sigmoid DPO, ref model on)."""
        device = self._setup_backend(backend)
        inputs = _make_inputs(n_pairs, T, H, V, dtype, device, with_ref=True, prompt_len=T // 3)

        test_out = dpo_loss(*inputs)
        ref_out = _reference_dpo_loss(*inputs)
        _assert_outputs_match(test_out, ref_out, dtype)

    @pytest.mark.parametrize("loss_type", ["hinge", "ipo", "robust"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_rejects_non_sigmoid(self, loss_type, backend, monkeypatch):
        """Only the sigmoid loss is supported; other variants fail loudly."""
        device = self._setup_backend(backend)
        inputs = _make_inputs(2, 8, 64, 128, torch.float32, device, with_ref=True, prompt_len=2)
        with pytest.raises(ValueError, match="sigmoid"):
            dpo_loss(*inputs, loss_type=loss_type)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_no_ref_model(self, backend, monkeypatch):
        """use_ref_model=False treats reference log-probs as zero."""
        device = self._setup_backend(backend)
        n_pairs, T, H, V = 2, 8, 64, 128
        dtype = torch.float32
        input_data, weight, target, _, _, _, _ = _make_inputs(n_pairs, T, H, V, dtype, device, with_ref=False)

        test_out = dpo_loss(input_data, weight, target, use_ref_model=False)
        ref_out = _reference_dpo_loss(input_data, weight, target, use_ref_model=False)
        _assert_outputs_match(test_out, ref_out, dtype)

    @pytest.mark.parametrize("average_log_prob", [False, True])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_nll_and_averaging(self, average_log_prob, backend, monkeypatch):
        """compute_nll_loss with alpha weighting, and average_log_prob."""
        device = self._setup_backend(backend)
        n_pairs, T, H, V = 2, 8, 64, 128
        dtype = torch.float32
        inputs = _make_inputs(n_pairs, T, H, V, dtype, device, with_ref=True, prompt_len=3)
        kwargs = {"compute_nll_loss": True, "alpha": 0.5, "average_log_prob": average_log_prob}

        test_out = dpo_loss(*inputs, **kwargs)
        ref_out = _reference_dpo_loss(*inputs, **kwargs)
        _assert_outputs_match(test_out, ref_out, dtype)
        assert test_out[5].abs() > 0, "nll_loss should be non-zero when compute_nll_loss=True"

    @pytest.mark.parametrize("backend", _backends)
    def test_op_bias(self, backend, monkeypatch):
        """Projection bias on both the policy and the reference model."""
        device = self._setup_backend(backend)
        n_pairs, T, H, V = 2, 8, 64, 128
        dtype = torch.float32
        inputs = _make_inputs(n_pairs, T, H, V, dtype, device, with_bias=True, with_ref=True)

        test_out = dpo_loss(*inputs)
        ref_out = _reference_dpo_loss(*inputs)
        _assert_outputs_match(test_out, ref_out, dtype)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, None])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_chunk_sizes(self, chunk_size, backend, monkeypatch):
        """Forced chunking (including a non-divisor chunk count) never changes
        results relative to the reference or the single-pass path."""
        device = self._setup_backend(backend)
        n_pairs, T, H, V = 4, 8, 64, 128
        dtype = torch.float32
        input_data, weight, target, bias, ref_input, ref_weight, ref_bias = _make_inputs(
            n_pairs, T, H, V, dtype, device, with_bias=True, with_ref=True, prompt_len=2
        )

        x_test = input_data.clone().requires_grad_(True)
        w_test = weight.clone().requires_grad_(True)
        out_test = dpo_loss(x_test, w_test, target, bias, ref_input, ref_weight, ref_bias, chunk_size=chunk_size)
        out_test[0].backward()

        x_ref = input_data.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        out_ref = _reference_dpo_loss(x_ref, w_ref, target, bias, ref_input, ref_weight, ref_bias)
        out_ref[0].backward()

        _assert_outputs_match(out_test, out_ref, dtype, what=f"[chunk={chunk_size}] ")
        for name, t, r in (("dInput", x_test.grad, x_ref.grad), ("dWeight", w_test.grad, w_ref.grad)):
            assert torch.allclose(t.float(), r.float(), **_tols(dtype)), (
                f"[chunk={chunk_size}] {name} mismatch: max={(t.float() - r.float()).abs().max().item():.6f}"
            )

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.bfloat16],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, dtype, backend, monkeypatch):
        """Gradients w.r.t. input, weight, and bias match the reference autograd."""
        device = self._setup_backend(backend)
        n_pairs, T, H, V = 2, 8, 64, 128
        input_data, weight, target, bias, ref_input, ref_weight, ref_bias = _make_inputs(
            n_pairs, T, H, V, dtype, device, with_bias=True, with_ref=True, prompt_len=2
        )

        x_test = input_data.clone().requires_grad_(True)
        w_test = weight.clone().requires_grad_(True)
        b_test = bias.clone().requires_grad_(True)
        out_test = dpo_loss(x_test, w_test, target, b_test, ref_input, ref_weight, ref_bias)
        out_test[0].backward()

        x_ref = input_data.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        b_ref = bias.clone().requires_grad_(True)
        out_ref = _reference_dpo_loss(x_ref, w_ref, target, b_ref, ref_input, ref_weight, ref_bias)
        out_ref[0].backward()

        for name, t, r in (
            ("dInput", x_test.grad, x_ref.grad),
            ("dWeight", w_test.grad, w_ref.grad),
            ("dBias", b_test.grad, b_ref.grad),
        ):
            assert t is not None, f"{name} is None"
            assert torch.allclose(t.float(), r.float(), **_tols(dtype)), (
                f"{name} mismatch: max={(t.float() - r.float()).abs().max().item():.6f}"
            )
