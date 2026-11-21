# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from families.embeddinggemma2.config import Config


def tiny_config():
    return {
        "model_type": "embedding_gemma2",
        "text_config": {
            "hidden_size": 4,
            "vocab_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 8,
            "hidden_size_per_layer_input": 4,
            "embedding_dim": 6,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "attention_bias": False,
            "sliding_window": 1,
            "max_position_embeddings": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "per_layer_config": {"01": {"head_dim": 4, "num_key_value_heads": 1}},
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            },
        },
    }


def test_layer_specific_dimensions_and_inclusive_radius():
    config = Config.from_dict(tiny_config())
    assert config.layer(0) == (2, 1, 10000.0, 1)
    assert config.layer(1) == (4, 1, 1000000.0, None)
    assert config.embedding_dim == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_dim", 3),
        ("num_key_value_heads", 3),
        ("rms_norm_eps", 0),
        ("layer_types", ["full_attention"]),
        ("hidden_activation", "silu"),
        ("attention_bias", True),
        ("vocab_size", 0),
        ("num_hidden_layers", True),
    ],
)
def test_reject_invalid_or_unsupported_text_config(field, value):
    raw = tiny_config()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_reject_non_default_rope_and_out_of_range_layer_override():
    raw = tiny_config()
    raw["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "linear"
    with pytest.raises(ValueError):
        Config.from_dict(raw)
    raw = tiny_config()
    raw["text_config"]["per_layer_config"]["02"] = {"head_dim": 4}
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_configuration_does_not_mutate_checkpoint_data():
    raw = tiny_config()
    before = copy.deepcopy(raw)
    Config.from_dict(raw)
    assert raw == before


def test_missing_per_layer_overrides_use_official_global_defaults():
    raw = tiny_config()
    del raw["text_config"]["per_layer_config"]
    assert Config.from_dict(raw).layer(1)[:2] == (512, 1)
