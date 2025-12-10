#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from tilegym import set_backend
from .. import common


class Test_MOE(common.PyTestCase):
    @staticmethod
    def reference(hidden_states, w1, w2, topk_weights, topk_ids):
        """
        Pure PyTorch reference implementation for MoE operation.
        This directly computes expert outputs and applies routing weights.
        """
        import torch.nn.functional as F

        device = hidden_states.device
        dtype = hidden_states.dtype
        num_tokens, hidden_size = hidden_states.shape
        n_experts, intermediate_size_x2, _ = w1.shape
        intermediate_size = intermediate_size_x2 // 2
        _, output_size, _ = w2.shape
        top_k = topk_ids.shape[1]

        # Split w1 into gate and up projections
        w1_gate = w1[:, :intermediate_size, :]  # [n_experts, intermediate_size, hidden_size]
        w1_up = w1[:, intermediate_size:, :]  # [n_experts, intermediate_size, hidden_size]

        # Create output tensor
        final_output = torch.zeros(num_tokens, output_size, dtype=dtype, device=device)

        # Process each token
        for token_idx in range(num_tokens):
            token_output = torch.zeros(output_size, dtype=dtype, device=device)

            # Process each selected expert for this token
            for k_idx in range(top_k):
                expert_idx = topk_ids[token_idx, k_idx].item()
                weight = topk_weights[token_idx, k_idx]

                # Gate and up projections
                gate_output = torch.matmul(hidden_states[token_idx], w1_gate[expert_idx].T)
                up_output = torch.matmul(hidden_states[token_idx], w1_up[expert_idx].T)

                # SiLU activation on gate, then multiply with up
                intermediate = F.silu(gate_output) * up_output

                # Down projection
                expert_output = torch.matmul(intermediate, w2[expert_idx].T)

                # Apply routing weight and accumulate
                token_output += expert_output * weight

            final_output[token_idx] = token_output

        return final_output

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "num_tokens, hidden_size, moe_intermediate_size, n_experts, top_k",
        [
            (16, 512, 256, 8, 2),
            (32, 1024, 512, 16, 4),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        num_tokens,
        hidden_size,
        moe_intermediate_size,
        n_experts,
        top_k,
        dtype,
        backend,
        arch,
    ):
        """Test correctness of MoE kernel against pure PyTorch reference implementation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if arch == "sm80" and dtype == torch.float8_e4m3fn:
            pytest.skip("Skip on sm80: don't support float8.")
        if backend == "cutile" and dtype == torch.float8_e4m3fn:
            pytest.skip("cutile haven't support it")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Failed to set backend {backend}: {e}")

        self.setUp()

        device = "cuda"

        init_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=init_dtype, device=device
        ).normal_(0, 0.5)

        # Create expert weights: w1 for gate+up projection, w2 for down projection
        w1 = torch.randn(
            n_experts,
            moe_intermediate_size * 2,  # Concatenated gate and up weights
            hidden_size,
            dtype=init_dtype,
            device=device,
        ).normal_(0, 0.1)
        w2 = torch.randn(
            n_experts,
            hidden_size,
            moe_intermediate_size,
            dtype=init_dtype,
            device=device,
        ).normal_(0, 0.1)

        quant_block = 128
        if dtype == torch.float8_e4m3fn:
            hidden_states_scale = 1.5 * torch.ones(
                (num_tokens, hidden_size // quant_block), device=device, dtype=torch.float32
            )
            w1_scale = 1.2 * torch.ones(
                (n_experts, moe_intermediate_size * 2 // quant_block, hidden_size // quant_block),
                device=device,
                dtype=torch.float32,
            )
            w2_scale = 0.5 * torch.ones(
                (n_experts, hidden_size // quant_block, moe_intermediate_size // quant_block),
                device=device,
                dtype=torch.float32,
            )
            hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
            w1_fp8 = w1.to(torch.float8_e4m3fn)
            w2_fp8 = w2.to(torch.float8_e4m3fn)

            hidden_states = hidden_states_fp8.to(init_dtype) * 1.5
            w1 = w1_fp8.to(init_dtype) * 1.2
            w2 = w2_fp8.to(init_dtype) * 0.5

        # Create topk_ids and topk_weights
        topk_ids = torch.randint(0, n_experts, (num_tokens, top_k), dtype=torch.long, device=device)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1).to(
            init_dtype
        )

        # Ensure all tensors are contiguous
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()

        # Define wrapper for fused_moe_kernel_interface
        from tilegym.ops.moe_interface import fused_experts_impl
        from tilegym.ops.moe_interface import fused_moe_kernel_interface

        def moe_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
            return fused_moe_kernel_interface(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
            )

        def moe_wrapper_fp8(hidden_states, w1, w2, topk_weights, topk_ids):
            kernel_configs = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": quant_block,
                "BLOCK_SIZE_K": quant_block,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 4,
            }

            return fused_experts_impl(
                hidden_states_fp8,
                w1_fp8,
                w2_fp8,
                topk_weights,
                topk_ids,
                inplace=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=hidden_states_scale,
                config=kernel_configs,
            )

        # Set tolerances based on dtype
        if dtype == torch.float16:
            rtol, atol = 5e-2, 5e-2
        elif dtype == torch.bfloat16:
            rtol, atol = 1e-1, 1e-1
        elif dtype == torch.float8_e4m3fn:
            rtol, atol = 5e-1, 5e-1
        else:
            rtol, atol = 1e-2, 1e-2

        # Use assertCorrectness to compare the implementations
        self.assertCorrectness(
            moe_wrapper_fp8 if dtype == torch.float8_e4m3fn else moe_wrapper,
            self.reference,
            {
                'hidden_states': hidden_states,
                'w1': w1,
                'w2': w2,
                'topk_weights': topk_weights,
                'topk_ids': topk_ids,
            },
            rtol=rtol,
            atol=atol,
        )
