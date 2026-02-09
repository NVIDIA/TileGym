# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class PartiallyFusedSwiGLUMLP(nn.Module):
    """
    Partially fused SwiGLU MLP that combines gate_proj + up_proj into a single linear layer,
    then uses fused silu_and_mul operation.

    This provides a middle ground between full fusion and standard implementation:
    - Step 1: Fused linear for gate_proj + up_proj (1 kernel instead of 2)
    - Step 2: Fused SiLU + multiplication (1 kernel instead of 2)
    - Step 3: Standard down_proj (1 kernel)

    Total: 3 kernels vs 5 in standard implementation

    This replaces:
        gate = gate_proj(x)     # kernel 1
        up = up_proj(x)         # kernel 2
        activated = silu(gate)  # kernel 3
        multiplied = activated * up  # kernel 4
        down_proj(multiplied)   # kernel 5

    With:
        fused_out = fused_gate_up_proj(x)  # kernel 1 (combines gate+up)
        glu_out = silu_and_mul(fused_out)  # kernel 2 (combines silu+mul)
        down_proj(glu_out)                 # kernel 3
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        # Keep individual weights for checkpoint compatibility
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Create fused weight parameter for gate+up projections
        # This will be initialized in _initialize_fused_weights()
        self.register_buffer("fused_gate_up_weight", None)

        # Validate activation function
        if hasattr(config, "hidden_act") and config.hidden_act not in [
            "silu",
            "swish",
        ]:
            raise ValueError(
                f"Activation function {config.hidden_act} not supported. "
                f"PartiallyFusedSwiGLUMLP only supports 'silu' or 'swish'."
            )

    def _initialize_fused_weights(self):
        """Initialize the fused weight from individual gate_proj and up_proj weights."""
        with torch.no_grad():
            # Concatenate gate_proj.weight and up_proj.weight along output dimension
            # gate_proj.weight: [intermediate_size, hidden_size]
            # up_proj.weight: [intermediate_size, hidden_size]
            # fused_weight: [2 * intermediate_size, hidden_size]
            self.fused_gate_up_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)

    def forward(self, x, use_torch_matmul=None):
        """
        Forward pass with optional torch.matmul fallback for training.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            use_torch_matmul: If True, use torch.matmul (supports backward).
                              If None, auto-detect based on requires_grad.
        """
        # Lazy initialize fused weights if needed
        if self.fused_gate_up_weight is None:
            self._initialize_fused_weights()

        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        # Auto-detect: use torch.matmul if backward is needed
        if use_torch_matmul is None:
            use_torch_matmul = x.requires_grad

        # Choose matmul function based on training mode
        matmul_fn = self.apply_matmul if use_torch_matmul else self.apply_matmul_internal

        # Lazy import to avoid circular dependency
        from tilegym.ops import silu_and_mul

        # Step 1: Fused gate+up projection
        # x @ fused_weight.T
        fused_output = matmul_fn(x, self.fused_gate_up_weight, trans_b=True)

        # Step 2: Fused SiLU and multiply
        # silu(fused_output[:, :intermediate_size]) * fused_output[:, intermediate_size:]
        # -> [batch, seq, intermediate_size]
        glu_output = silu_and_mul(fused_output)

        # Step 3: Down projection
        # glu_output @ down_proj.weight.T
        result = matmul_fn(glu_output, self.down_proj.weight, trans_b=True)

        return result.view(*orig_shape)

    def apply_matmul(self, x, weight, trans_b):
        return torch.matmul(x, weight.T if trans_b else weight)

    def apply_matmul_internal(self, x, weight, trans_b):
        from tilegym.ops import matmul

        return matmul(x, weight, trans_b=trans_b, use_tma=True, static_persistent=True)

    def update_fused_weights(self):
        """
        Update fused weights when individual weights change.
        Call this after loading checkpoints or updating weights.
        """
        self._initialize_fused_weights()
