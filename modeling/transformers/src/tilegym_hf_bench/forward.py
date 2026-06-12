# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


class NaiveForwardWrapper:
    def __init__(
        self,
        model,
        tokenizer,
        messages_list,
        args,
        device="cuda",
        **default_kwargs,
    ):
        self.model = model
        self.default_kwargs = default_kwargs
        self.tokenizer = tokenizer
        self.tokenized_inputs = tokenizer(messages_list, return_tensors="pt").to(device)
        self.input_seq_len = self.tokenized_inputs.input_ids.shape[1]
        self.default_kwargs.update(
            {
                "input_ids": self.tokenized_inputs.input_ids,
                "attention_mask": self.tokenized_inputs.attention_mask,
                "max_new_tokens": args.output_length,
                "min_new_tokens": args.output_length,
            }
        )

    def get_input_seq_len(self):
        return self.input_seq_len

    def update_params(self, **kwargs):
        self.default_kwargs.update(kwargs)

    def forward(self):
        return self.model.generate(**self.default_kwargs)

    def post_process(self, outputs):
        return outputs
