# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Correctness tests for the LFM2-MoE fused cuTile causal conv1d kernels.

These wrappers are not backend-dispatched ops, so they are exercised directly
(rather than via the ``PyTestCase`` harness) against a pure-torch reference that
matches ``Lfm2MoeShortConv``'s stock semantics. Requires a CUDA device with
cuTile; skipped otherwise.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="LFM2 conv kernels require CUDA/cuTile")


def _ref_prefill(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """nn.Conv1d(groups=D, padding=K-1)(x)[..., :L] — depthwise causal conv."""
    B, D, L = x.shape
    K = weight.shape[1]
    xp = F.pad(x, (K - 1, 0))
    w = weight.view(D, 1, K)
    return F.conv1d(xp, w, groups=D)


def _ref_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor):
    """Roll the K-wide window and convolve; returns (out, rolled_state).

    Mirrors HF's fallback ``causal_conv1d_update`` on the single-token decode
    path: ``x`` is 3D ``(B, D, 1)`` and the output keeps that rank.

    The multiply-accumulate is done in float32 to match the kernel under test
    (which upcasts to f32 internally, like the CUDA causal-conv1d package) --
    a pure-bf16 reference rounds each product to bf16 before summing and is
    *less* accurate than the kernel, showing up as spurious 1-2 ulp diffs.
    """
    rolled = torch.cat([conv_state[..., 1:], x], dim=-1)
    out = (rolled.float() * weight.float().unsqueeze(0)).sum(-1, keepdim=True).to(x.dtype)
    return out, rolled


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("D, L", [(2048, 16), (2048, 128), (256, 7), (300, 33)])
def test_prefill(D, L, dtype):
    from tilegym.transformers.lfm2_moe.kernels.causal_conv1d_prefill import lfm2_causal_conv1d_fn_cutile

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(1, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, 3, dtype=dtype, device=device)

    out = lfm2_causal_conv1d_fn_cutile(x, weight)
    ref = _ref_prefill(x, weight)

    assert out.shape == ref.shape == (1, D, L)
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("D", [2048, 256, 300])
def test_update(D, dtype):
    """Exercise the kernel with the exact 3D call contract of
    ``Lfm2MoeShortConv.forward``'s cached-decode branch: x is ``(B, D, 1)``
    and the output must be ``(B, D, 1)`` (it is multiplied with the 3D gate
    ``C`` at the call site)."""
    from tilegym.transformers.lfm2_moe.kernels.causal_conv1d_update import lfm2_causal_conv1d_update_cutile

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(1, D, 1, dtype=dtype, device=device)
    conv_state = torch.randn(1, D, 3, dtype=dtype, device=device)
    weight = torch.randn(D, 3, dtype=dtype, device=device)

    ref_out, ref_state = _ref_update(x, conv_state, weight)

    cs = conv_state.clone()
    out = lfm2_causal_conv1d_update_cutile(x, cs, weight)

    assert out.shape == (1, D, 1)
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    # output matches reference
    torch.testing.assert_close(out.float(), ref_out.float(), atol=atol, rtol=1e-3)
    # conv_state was rolled in place to [s1, s2, x]
    torch.testing.assert_close(cs.float(), ref_state.float(), atol=atol, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_2d_legacy_call(dtype):
    """Older transformers (<= 5.10.x, `cuda_kernels_forward`) call the update
    with a 2D ``(B, D)`` input (``Bx.squeeze(-1)``) and unsqueeze the result
    at the call site. The wrapper must accept that convention too."""
    from tilegym.transformers.lfm2_moe.kernels.causal_conv1d_update import lfm2_causal_conv1d_update_cutile

    torch.manual_seed(0)
    device = torch.device("cuda")
    D = 512
    x = torch.randn(1, D, 1, dtype=dtype, device=device)
    conv_state = torch.randn(1, D, 3, dtype=dtype, device=device)
    weight = torch.randn(D, 3, dtype=dtype, device=device)

    ref_out, ref_state = _ref_update(x, conv_state, weight)

    cs = conv_state.clone()
    out = lfm2_causal_conv1d_update_cutile(x.squeeze(-1), cs, weight)  # 2D legacy call

    assert out.shape == (1, D)  # rank follows the input
    atol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref_out.squeeze(-1).float(), atol=atol, rtol=1e-3)
    torch.testing.assert_close(cs.float(), ref_state.float(), atol=atol, rtol=1e-3)


def test_prefill_matches_update_stepwise():
    """A full-sequence prefill should equal stepping the update kernel token by
    token from a zero-initialized K-wide state (end-to-end conv consistency)."""
    from tilegym.transformers.lfm2_moe.kernels.causal_conv1d_prefill import lfm2_causal_conv1d_fn_cutile
    from tilegym.transformers.lfm2_moe.kernels.causal_conv1d_update import lfm2_causal_conv1d_update_cutile

    torch.manual_seed(0)
    device = torch.device("cuda")
    D, L = 512, 12
    x = torch.randn(1, D, L, dtype=torch.float32, device=device)
    weight = torch.randn(D, 3, dtype=torch.float32, device=device)

    prefill = lfm2_causal_conv1d_fn_cutile(x, weight)  # (1, D, L)

    conv_state = torch.zeros(1, D, 3, dtype=torch.float32, device=device)
    step_outs = []
    for t in range(L):
        # Keep the 3D (1, D, 1) call shape used by the HF decode path.
        step_outs.append(lfm2_causal_conv1d_update_cutile(x[:, :, t : t + 1], conv_state, weight))
    stepwise = torch.cat(step_outs, dim=-1)  # (1, D, L)

    torch.testing.assert_close(prefill, stepwise, atol=1e-4, rtol=1e-3)
