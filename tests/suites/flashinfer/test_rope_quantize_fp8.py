# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional
from typing import Tuple
from typing import Union

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.flashinfer import ops as tilegym_flashinfer_ops


# reference implementation of RotaryEmbedding
class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.device = device
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=self.device)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, num_heads, head_size]
            cos: [num_tokens, head_size // 2]
            sin: [num_tokens, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        """
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)

        # Note: the is different from the vLLM's implementation,
        # We added float32 conversion because float32 is required for the rotary embedding to work correctly for long contexts
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        query = query.to(self.dtype)
        key = key.to(self.dtype)
        return query, key


class Test_FlashInfer_RopeQuantizeFp8(common.PyTestCase):
    @pytest.mark.parametrize("num_tokens", [1, 19, 128])
    @pytest.mark.parametrize("num_qo_heads", [128])
    @pytest.mark.parametrize("head_size", [576])
    @pytest.mark.parametrize("rotary_dim", [64])
    @pytest.mark.parametrize("max_position_embeddings", [4096])
    @pytest.mark.parametrize("base", [10000])
    @pytest.mark.parametrize("is_neox_style", [False])
    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @pytest.mark.parametrize(
        "framework",
        [
            "cutile",
        ],
    )
    def test_op(
        self,
        num_tokens,
        num_qo_heads,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        input_dtype,
        quant_dtype,
        framework,
        arch,
    ):
        _impl_fw = ["cutile"]
        if framework not in _impl_fw:
            pytest.skip(f"Framework {framework} not supported")
        if tilegym.is_backend_available(framework):
            tilegym.set_backend(framework)
        else:
            pytest.skip(f"Backend {framework} is not available")
        if framework == "cutile":
            pytest.xfail("CuTile rope_kernel: TileAS lowering to NVVM fails with invalid use-def chain ")

        if arch == "sm80" and "float8" in quant_dtype.__repr__():
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        device = "cuda:0"
        # Fixed seed for reproducibility across tests
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        q_in = torch.randn(num_tokens, num_qo_heads, 576, dtype=input_dtype, device=device)
        k_in = torch.randn(num_tokens, 576, dtype=input_dtype, device=device)
        pos_ids = torch.arange(num_tokens, device=device)

        # reference implementation
        rope_flashinfer = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            input_dtype,
            device,
        )
        q_out_f16_ref, k_out_f16_ref = rope_flashinfer.forward_native(pos_ids, q_in, k_in)
        q_out_f8_ref, k_out_f8_ref = map(
            lambda x: x.to(quant_dtype),
            (q_out_f16_ref, k_out_f16_ref),
        )

        # kernel implementation
        q_out = torch.empty_like(q_in, dtype=quant_dtype)
        k_out = torch.empty_like(k_in, dtype=quant_dtype)
        tilegym_flashinfer_ops.rope_quantize_fp8(
            q_in[..., :rotary_dim],
            k_in[..., :rotary_dim],
            q_in[..., rotary_dim:],
            k_in[..., rotary_dim:],
            rope_flashinfer.cos_sin_cache,
            pos_ids,
            is_neox=is_neox_style,
            q_rope_out=q_out[..., :rotary_dim],
            k_rope_out=k_out[..., :rotary_dim],
            q_nope_out=q_out[..., rotary_dim:],
            k_nope_out=k_out[..., rotary_dim:],
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        torch.testing.assert_close(q_out_f8_ref.float(), q_out.float(), atol=1e-2, rtol=2e-1)
        torch.testing.assert_close(k_out_f8_ref.float(), k_out.float(), atol=1e-2, rtol=2e-1)
