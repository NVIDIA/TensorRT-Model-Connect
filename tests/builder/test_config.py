# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config.py — ModelConfig parsing from HF config.json.

Trace: ARCH-CFG-002, UD-CFG-01
Intent: Validate ModelConfig parsing from HF config.json across supported schema shapes, including field aliases, VL text_config merge, and edge cases.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: Parsed fields (model_type, hidden_size, num_heads, etc.) match expected values for each generic config format.
"""

from __future__ import annotations

import json
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.config import ModelConfig


class TestModelConfigFromJson:
    def test_standard_decoder_keys(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "standard_decoder",
            "architectures": ["StandardDecoderForCausalLM"],
            "vocab_size": 151936,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
        }))
        assert cfg.model_type == "standard_decoder"
        assert cfg.vocab_size == 151936
        assert cfg.hidden_size == 1024
        assert cfg.intermediate_size == 3072
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 4
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 1000000.0

    def test_standard_decoder_with_kv_heads(self):
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "kv_decoder",
            "architectures": ["KVDecoderForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
        }))
        assert cfg.model_type == "kv_decoder"
        assert cfg.hidden_size == 2048
        assert cfg.num_key_value_heads == 4
        assert cfg.rms_norm_eps == 1e-5

    def test_n_embd_nonstandard_keys(self):
        """Some configs use n_embd, n_head, n_layer, n_inner."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "n_embd_decoder",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "vocab_size": 50257,
            "layer_norm_epsilon": 1e-5,
        }))
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.num_hidden_layers == 12
        assert cfg.intermediate_size == 3072
        assert cfg.rms_norm_eps == 1e-5

    def test_d_model_nonstandard_keys(self):
        """Some configs use d_model, attention_heads, num_layers."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "d_model_decoder",
            "d_model": 2560,
            "attention_heads": 32,
            "num_layers": 30,
            "vocab_size": 250880,
            "layer_norm_eps": 1e-5,
        }))
        assert cfg.hidden_size == 2560
        assert cfg.num_attention_heads == 32
        assert cfg.num_hidden_layers == 30
        assert cfg.rms_norm_eps == 1e-5
        # intermediate_size fallback: hidden * 4
        assert cfg.intermediate_size == 2560 * 4

    def test_layer_norm_eps_alias(self):
        """layer_norm_eps is accepted as an epsilon alias."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "layer_norm_eps_decoder",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "ffn_dim": 3072,
            "vocab_size": 50272,
            "layer_norm_eps": 1e-5,
        }))
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.intermediate_size == 3072

    def test_norm_epsilon_alias(self):
        """norm_epsilon is accepted as an epsilon alias."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "norm_epsilon_decoder",
            "hidden_size": 4544,
            "num_attention_heads": 71,
            "num_hidden_layers": 32,
            "vocab_size": 65024,
            "norm_epsilon": 1e-5,
        }))
        assert cfg.rms_norm_eps == 1e-5


class TestHeadDim:
    def test_computed(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16)
        assert cfg.head_dim == 64

    def test_explicit_override(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16, _head_dim=128)
        assert cfg.head_dim == 128

    def test_zero_heads(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=0)
        assert cfg.head_dim == 0


class TestAttentionSize:
    def test_basic(self):
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=16)
        assert cfg.attention_size == 1024  # 16 * 64

    def test_with_explicit_head_dim(self):
        cfg = ModelConfig(
            hidden_size=1024, num_attention_heads=16, _head_dim=128)
        assert cfg.attention_size == 2048  # 16 * 128


class TestFromDir:
    def test_from_dir(self, tmp_path):
        config = {
            "model_type": "test",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "vocab_size": 10000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.model_type == "test"
        assert cfg.hidden_size == 512


class TestEdgeCases:
    def test_missing_keys_defaults(self):
        cfg = ModelConfig.from_json("{}")
        assert cfg.model_type == ""
        assert cfg.hidden_size == 0
        assert cfg.num_attention_heads == 1  # fallback
        assert cfg.rms_norm_eps == 1e-5  # fallback

    def test_tie_word_embeddings(self):
        cfg = ModelConfig.from_json(json.dumps({
            "tie_word_embeddings": True,
            "hidden_size": 768,
            "num_attention_heads": 12,
        }))
        assert cfg.tie_word_embeddings is True

    def test_max_position_embeddings_n_positions(self):
        """Some models use n_positions instead of max_position_embeddings."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "n_positions": 1024,
        }))
        assert cfg.max_position_embeddings == 1024

    def test_hidden_act(self):
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "hidden_act": "silu",
        }))
        assert cfg.hidden_act == "silu"

    def test_activation_function_fallback(self):
        """Some models use activation_function instead of hidden_act."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "activation_function": "gelu_new",
        }))
        assert cfg.hidden_act == "gelu_new"

    def test_rope_theta_from_rope_parameters(self):
        """Some configs store rope_theta inside rope_parameters."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "rope_parameters_decoder",
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rope_parameters": {
                "rope_type": "long_context",
                "rope_theta": 500000.0,
            },
        }))
        assert cfg.rope_theta == 500000.0

    def test_rope_theta_top_level_takes_precedence(self):
        """Top-level rope_theta takes precedence over rope_parameters."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "rope_parameters_decoder",
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "rope_theta": 1000000.0,
            "rope_parameters": {
                "rope_theta": 500000.0,
            },
        }))
        assert cfg.rope_theta == 1000000.0

    def test_rope_theta_default_no_rope_parameters(self):
        """Default rope_theta when neither top-level nor rope_parameters present."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "rope_parameters_decoder",
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }))
        assert cfg.rope_theta == 10000.0

    def test_rope_theta_from_rope_scaling(self):
        """Some configs store rope_theta inside rope_scaling."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "rope_scaling_decoder",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "rope_scaling": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        }))
        assert cfg.rope_theta == 1000000.0

    def test_raw_dict_preserved(self):
        raw = {
            "model_type": "test",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "custom_field": "custom_value",
        }
        cfg = ModelConfig.from_json(json.dumps(raw))
        assert cfg.raw["custom_field"] == "custom_value"


class TestModelConfigExtended:
    """Extended edge case tests for ModelConfig parsing."""

    # --- VL text_config merging ---

    def test_text_config_merged_into_top_level(self):
        """VL models nest text model config under text_config; fields merge to top."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "vision_language_decoder",
            "architectures": ["VisionLanguageForConditionalGeneration"],
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "intermediate_size": 2816,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
            },
        }))
        assert cfg.model_type == "vision_language_decoder"
        assert cfg.hidden_size == 1024
        assert cfg.num_hidden_layers == 24
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 4
        assert cfg.intermediate_size == 2816
        assert cfg.vocab_size == 151936
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.rope_theta == 1000000.0

    def test_text_config_overrides_top_level(self):
        """text_config fields override top-level fields (merged second)."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "vision_language_decoder",
            "hidden_size": 512,
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
            },
        }))
        # text_config's hidden_size should win over top-level
        assert cfg.hidden_size == 1024

    def test_no_text_config_uses_top_level(self):
        """Without text_config, top-level fields are used directly."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "standard_decoder",
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
        }))
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 22

    def test_text_config_preserves_raw_as_original(self):
        """raw dict should be the original JSON dict, not the merged one."""
        original = {
            "model_type": "vision_language_decoder",
            "vision_config": {"image_size": 224},
            "text_config": {
                "hidden_size": 1024,
                "num_attention_heads": 16,
            },
        }
        cfg = ModelConfig.from_json(json.dumps(original))
        assert "text_config" in cfg.raw
        assert "vision_config" in cfg.raw
        assert cfg.raw["vision_config"]["image_size"] == 224

    # --- Error handling ---

    def test_malformed_json_raises(self):
        """Malformed JSON string raises json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            ModelConfig.from_json("{not valid json}")

    def test_missing_num_hidden_layers_defaults_to_zero(self):
        """Config without num_hidden_layers defaults to 0."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "test",
            "hidden_size": 768,
            "num_attention_heads": 12,
        }))
        assert cfg.num_hidden_layers == 0

    def test_rope_theta_explicit_none_falls_back_to_default(self):
        """Config with rope_theta=None falls back to default 10000.0."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "rope_parameters_decoder",
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "rope_theta": None,
        }))
        assert cfg.rope_theta == 10000.0

    # --- Edge cases ---

    def test_intermediate_size_takes_priority_over_n_inner(self):
        """When both intermediate_size and n_inner are present, intermediate_size wins."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "n_embd_decoder",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "intermediate_size": 4096,
            "n_inner": 3072,
        }))
        # intermediate_size is checked first in the `or` chain
        assert cfg.intermediate_size == 4096

    def test_n_inner_used_when_no_intermediate_size(self):
        """n_inner is used as fallback when intermediate_size is absent."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "n_embd_decoder",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "n_inner": 3072,
        }))
        assert cfg.intermediate_size == 3072

    def test_rope_scaling_with_nested_rope_type_and_factor(self):
        """Config with rope_scaling dict — verify raw dict captures it."""
        raw_config = {
            "model_type": "rope_scaling_decoder",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "long_context",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }
        cfg = ModelConfig.from_json(json.dumps(raw_config))
        assert cfg.rope_theta == 500000.0
        # rope_scaling should be preserved in raw dict
        assert "rope_scaling" in cfg.raw
        assert cfg.raw["rope_scaling"]["rope_type"] == "long_context"
        assert cfg.raw["rope_scaling"]["factor"] == 8.0

    def test_head_dim_from_config_json(self):
        """Explicit head_dim in config.json overrides computed value."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "head_dim_decoder",
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "head_dim": 96,
        }))
        # Without head_dim: 3072/32 = 96, but the explicit override path matters
        assert cfg.head_dim == 96
        assert cfg._head_dim == 96

    def test_intermediate_size_fallback_to_hidden_times_four(self):
        """When no intermediate_size, n_inner, or ffn_dim, falls back to hidden*4."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "test",
            "hidden_size": 512,
            "num_attention_heads": 8,
        }))
        assert cfg.intermediate_size == 512 * 4

    def test_empty_string_not_valid_json(self):
        """Empty string raises json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            ModelConfig.from_json("")

    def test_num_key_value_heads_defaults_to_num_heads(self):
        """When num_key_value_heads is missing, defaults to num_attention_heads."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "kv_decoder",
            "hidden_size": 1024,
            "num_attention_heads": 16,
        }))
        assert cfg.num_key_value_heads == 16


class TestModelConfigEdgeCases:
    """Negative and edge-case tests documenting ModelConfig behavior with bad input.

    ModelConfig is a plain dataclass with no validation — it stores whatever
    values are passed. These tests document that behavior explicitly.
    """

    @pytest.mark.parametrize("field_name,value", [
        ("hidden_size", -1),
        ("hidden_size", -1024),
        ("num_attention_heads", -1),
        ("num_attention_heads", -16),
        ("num_hidden_layers", -1),
        ("num_hidden_layers", -28),
    ])
    def test_negative_dimensions_stored_as_is(self, field_name, value):
        """ModelConfig stores negative dimensions without validation."""
        cfg = ModelConfig(**{field_name: value})
        assert getattr(cfg, field_name) == value

    @pytest.mark.parametrize("field_name,value", [
        ("hidden_size", -512),
        ("num_attention_heads", -8),
        ("num_hidden_layers", -4),
    ])
    def test_negative_dimensions_via_from_json(self, field_name, value):
        """from_json stores negative dimensions from JSON without validation."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "test",
            field_name: value,
        }))
        assert getattr(cfg, field_name) == value

    def test_zero_layers(self):
        """num_hidden_layers=0 is stored and produces a valid config."""
        cfg = ModelConfig(
            hidden_size=1024, num_attention_heads=16, num_hidden_layers=0)
        assert cfg.num_hidden_layers == 0
        # head_dim and attention_size still compute correctly
        assert cfg.head_dim == 64
        assert cfg.attention_size == 1024

    def test_zero_layers_via_from_json(self):
        """from_json with num_hidden_layers=0 produces a zero-layer config."""
        cfg = ModelConfig.from_json(json.dumps({
            "model_type": "test",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 0,
        }))
        assert cfg.num_hidden_layers == 0

    def test_head_dim_with_negative_num_heads(self):
        """head_dim returns 0 when num_attention_heads is negative (<=0 guard)."""
        cfg = ModelConfig(hidden_size=1024, num_attention_heads=-4)
        assert cfg.head_dim == 0
        assert cfg.attention_size == 0

    def test_string_hidden_size_via_constructor(self):
        """Passing string where int expected — dataclass stores it without type check."""
        cfg = ModelConfig(hidden_size="abc")  # type: ignore[arg-type]
        assert cfg.hidden_size == "abc"

    def test_float_num_hidden_layers_via_constructor(self):
        """Passing float where int expected — dataclass stores it without type check."""
        cfg = ModelConfig(num_hidden_layers=2.5)  # type: ignore[arg-type]
        assert cfg.num_hidden_layers == 2.5

    def test_string_hidden_size_from_json(self):
        """JSON string value for hidden_size — from_json stores it (no int coercion)."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": "abc",
            "num_attention_heads": 12,
        }))
        # The `or` chain in from_json: d.get("hidden_size", 0) returns "abc"
        # (truthy string), so hidden_size = "abc"
        assert cfg.hidden_size == "abc"

    def test_float_num_hidden_layers_from_json(self):
        """JSON float value for num_hidden_layers — from_json stores it as float."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 2.5,
        }))
        assert cfg.num_hidden_layers == 2.5

    def test_none_hidden_size_from_json(self):
        """JSON null for hidden_size — falls through or-chain to 0."""
        cfg = ModelConfig.from_json(json.dumps({
            "hidden_size": None,
            "num_attention_heads": 12,
        }))
        # None is falsy, so the `or` chain falls through: 0 or 0 or ... = 0
        assert cfg.hidden_size == 0
