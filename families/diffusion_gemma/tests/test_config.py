# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from families.diffusion_gemma.config import Config


def checkpoint_config():
    return json.loads((Path(__file__).parent / "data/diffusiongemma-26b-a4b-config.json").read_text())


def test_checkpoint_global_and_sliding_attention_have_distinct_shapes():
    config = Config.from_dict(checkpoint_config())
    assert config.canvas_length == 256
    assert config.layer_dimensions(0) == (8, 256)
    assert config.layer_dimensions(5) == (2, 512)
    sliding, full = config.layer_weight_shapes(0), config.layer_weight_shapes(5)
    assert sliding["self_attn.v_proj.weight"] == (2048, 2816)
    assert "self_attn.v_proj.weight" not in full
    assert full["self_attn.k_proj.weight"] == (1024, 2816)
    assert full["experts.gate_up_proj"] == (128, 1408, 2816)
    assert full["experts.down_proj"] == (128, 2816, 704)
    assert config.embedding_weight_shapes()["self_conditioning.pre_norm.weight"] == (2816,)


@pytest.mark.parametrize(("key", "value"), [
    ("num_key_value_heads", 3), ("top_k_experts", 129), ("num_experts", 0),
    ("attention_bias", True), ("hidden_activation", "silu"),
    ("rms_norm_eps", float("nan")), ("final_logit_softcapping", 0),
    ("use_bidirectional_attention", "all"),
])
def test_unsupported_mathematical_contracts_are_rejected(key, value):
    raw = checkpoint_config()
    raw["text_config"][key] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_full_attention_rotary_fraction_must_preserve_pairs():
    raw = checkpoint_config()
    raw["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 3 / 512
    with pytest.raises(ValueError, match="rotary dimension"):
        Config.from_dict(raw)


def test_causal_gemma_checkpoint_is_not_a_diffusion_model():
    raw = checkpoint_config()
    raw["model_type"] = "gemma4"
    with pytest.raises(ValueError, match="diffusion_gemma checkpoint"):
        Config.from_dict(raw)
