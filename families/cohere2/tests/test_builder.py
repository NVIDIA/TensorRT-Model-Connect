# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology and checkpoint contracts, without a GPU or downloaded assets."""

from dataclasses import replace

import numpy as np
import pytest

from families.cohere2.config import parse_config
from families.cohere2.model import tokenizer_contract
from families.cohere2.weights import load_weights


def small_config():
    return dict(
        model_type="cohere2",
        architectures=["Cohere2ForCausalLM"],
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        rope_theta=50000,
        use_embedding_sharing=True,
        use_parallel_block=True,
        position_embedding_type="rope_gptj",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_qk_norm", True),
        ("use_parallel_block", False),
        ("attention_bias", True),
        ("position_embedding_type", "rope"),
        ("rotary_pct", 0.5),
        ("head_dim", 3),
        ("num_key_value_heads", 3),
        ("layer_types", ["full_attention"]),
        ("sliding_window", 0),
        ("rope_scaling", {"factor": 2}),
        ("logit_scale", float("nan")),
        ("eos_token_id", -1),
        ("hidden_size", True),
    ],
)
def test_unsupported_topologies_are_rejected(field, value):
    with pytest.raises(ValueError):
        parse_config({**small_config(), field: value})


def test_bos_template_is_not_silently_changed():
    template = {
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<BOS_TOKEN>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "special_tokens": {"<BOS_TOKEN>": {"ids": [2]}},
        }
    }
    tokenizer_contract(template, 2)
    with pytest.raises(ValueError):
        tokenizer_contract(template, 1)
    template["post_processor"]["single"].reverse()
    with pytest.raises(ValueError):
        tokenizer_contract(template, 2)


def test_original_checkpoint_weights_and_tying(tmp_path):
    import torch
    from transformers import Cohere2Config, Cohere2ForCausalLM

    raw = small_config()
    config = Cohere2Config(**raw)
    model = Cohere2ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(tmp_path, safe_serialization=True)
    parsed = parse_config(raw)
    loaded = load_weights(tmp_path, parsed)
    assert loaded["lm_head.weight"] is loaded["model.embed_tokens.weight"]
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(loaded[name], tensor.float().numpy())
    with pytest.raises(ValueError, match="expected"):
        load_weights(tmp_path, replace(parsed, vocab_size=33))
