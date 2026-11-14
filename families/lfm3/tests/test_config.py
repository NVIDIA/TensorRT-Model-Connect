# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from families.lfm3.config import validate_dense_lfm3_config


def config_raw():
    return dict(
        model_type="lfm3",
        architectures=["Lfm3ForCausalLM"],
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention", "sliding_attention"],
        conv_L_cache=7,
        conv_bias=False,
        norm_eps=1e-5,
        rope_parameters={
            "full_attention": dict(rope_type="default", rope_theta=1e6, partial_rotary_factor=0.5),
            "sliding_attention": dict(rope_type="default", rope_theta=1e4),
        },
        sliding_window=256,
        use_sandwich_norm=True,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )


def test_three_layer_types_keep_direct_mlp_width_and_independent_rope():
    cfg = validate_dense_lfm3_config(SimpleNamespace(raw=config_raw()))
    assert cfg.intermediate_size == 16
    assert cfg.num_attention_layers == 2
    assert cfg.num_conv_layers == 1
    assert cfg.conv_l_cache == 7
    assert cfg.rope_spec("full_attention") == (2, 1e6)
    assert cfg.rope_spec("sliding_attention") == (4, 1e4)
    assert cfg.bos_token_id == 0
    assert cfg.sliding_window == 256
    assert cfg.use_sandwich_norm


@pytest.mark.parametrize(
    "name,value",
    [
        ("hidden_size", True),
        ("head_dim", 3.5),
        ("sliding_window", 0),
        ("num_key_value_heads", 3),
        ("bos_token_id", 16),
        ("eos_token_id", -1),
        ("quantization_config", {"quant_method": "gptq"}),
    ],
)
def test_invalid_geometry_and_ids_fail_before_loading_weights(name, value):
    raw = config_raw()
    raw[name] = value
    with pytest.raises(ValueError):
        validate_dense_lfm3_config(SimpleNamespace(raw=raw))
