# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import BatchEncoding

from families.xglm.tests.test_e2e import _render_prompt


class Tokenizer:
    eos_token = "</s>"
    eos_token_id = 9

    def __call__(self, prompt, **kwargs):
        # HF appends </s> even for a continuation prompt; retain XGLM's prefix.
        ids = [2] + [4, 5]
        if prompt.endswith(self.eos_token):
            ids.append(self.eos_token_id)
        ids.append(self.eos_token_id)
        tensor = torch.tensor([ids])
        return BatchEncoding({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})


@pytest.mark.parametrize("explicit_eos", [False, True])
def test_reference_prompt_matches_native_continuation_ids(explicit_eos):
    tokenizer = Tokenizer()
    prompt = "The capital of France is" + (tokenizer.eos_token if explicit_eos else "")
    inputs = _render_prompt(tokenizer, prompt, {})
    expected = [2] + [4, 5] + ([9] if explicit_eos else [])
    assert inputs["input_ids"].tolist() == [expected]
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape
    assert inputs.to("cpu") is inputs
