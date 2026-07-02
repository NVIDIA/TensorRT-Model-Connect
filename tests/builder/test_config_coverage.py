# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-focused tests for merge/fallback behavior in ModelConfig.

Trace: ARCH-CFG-002, UD-CFG-02
Intent: Validate ModelConfig merge and fallback paths including language_config merge for VL models, text_config promotion, and head_dim computation edge cases.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: Nested config sections (language_config, text_config) merge correctly, guard conditions prevent unintended overrides, and derived fields compute accurately.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.config import ModelConfig


def test_language_config_merges_when_hidden_size_missing():
    """Intent: cover language_config merge path.
    Preconditions: top-level hidden_size is absent and language_config contains decoder fields.
    Postconditions: decoder fields come from language_config while top-level identity fields are preserved.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "language_config_vl",
        "architectures": ["LanguageConfigVLForCausalLM"],
        "vision_config": {"image_size": 384},
        "language_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 152064,
        },
    }))

    assert cfg.model_type == "language_config_vl"
    assert cfg.architectures == ["LanguageConfigVLForCausalLM"]
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 30
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.intermediate_size == 11008
    assert cfg.vocab_size == 152064
    assert cfg.raw["vision_config"]["image_size"] == 384


def test_language_config_is_skipped_when_top_level_hidden_size_exists():
    """Intent: validate language_config guard condition.
    Preconditions: top-level hidden_size is already present and language_config is also present.
    Postconditions: top-level hidden_size is retained and no language_config override occurs.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "language_config_vl",
        "hidden_size": 1024,
        "language_config": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
        },
    }))

    assert cfg.hidden_size == 1024


def test_text_config_merge_still_works_without_top_level_identity_fields():
    """Intent: cover text_config merge branches where model_type/architectures are absent.
    Preconditions: text_config is present but top-level model_type and architectures are missing.
    Postconditions: text fields are merged and identity fields remain at their default values.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
        },
    }))

    assert cfg.model_type == ""
    assert cfg.architectures == []
    assert cfg.hidden_size == 1536
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 12


def test_llm_config_merges_when_hidden_size_missing():
    """Intent: cover llm_config merge path.
    Preconditions: hidden_size is missing at top-level and llm_config is a dict.
    Postconditions: LLM fields are loaded from llm_config and top-level vision_config is preserved.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "llm_config_vl",
        "architectures": ["LLMConfigVLForConditionalGeneration"],
        "vision_config": {"image_size": 448},
        "llm_config": {
            "hidden_size": 3072,
            "num_hidden_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
        },
    }))

    assert cfg.model_type == "llm_config_vl"
    assert cfg.hidden_size == 3072
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 24
    assert cfg.num_key_value_heads == 8
    assert cfg.raw["vision_config"]["image_size"] == 448


def test_llm_config_non_dict_is_ignored():
    """Intent: cover llm_config type guard false branch.
    Preconditions: hidden_size is missing and llm_config is present but not a dict.
    Postconditions: llm_config is ignored and hidden_size remains unresolved (zero).
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "llm_config_vl",
        "llm_config": ["not", "a", "dict"],
    }))

    assert cfg.hidden_size == 0


def test_thinker_text_config_merges_and_propagates_thinker_vision_config():
    """Intent: cover thinker_config.text_config merge behavior.
    Preconditions: hidden_size is absent, thinker_config.text_config is present, and merged config lacks vision_config.
    Postconditions: text fields are merged and thinker_config vision_config is propagated.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "thinker_text_config",
        "architectures": ["ThinkerTextConfigForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 20,
                "num_attention_heads": 16,
            },
            "vision_config": {"image_size": 560},
        },
    }))

    assert cfg.model_type == "thinker_text_config"
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 20
    assert cfg.num_attention_heads == 16
    assert cfg.raw["thinker_config"]["vision_config"]["image_size"] == 560


def test_thinker_text_config_non_dict_is_ignored():
    """Intent: cover thinker_config.text_config type guard false branch.
    Preconditions: hidden_size is missing and thinker_config.text_config is not a dict.
    Postconditions: thinker text merge is skipped and hidden_size remains unresolved (zero).
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "thinker_text_config",
        "thinker_config": {
            "text_config": ["not", "a", "dict"],
        },
    }))

    assert cfg.hidden_size == 0


def test_thinker_vision_config_does_not_overwrite_existing_top_level_value():
    """Intent: verify vision_config precedence in thinker merge.
    Preconditions: top-level vision_config exists and thinker_config also carries vision_config.
    Postconditions: top-level vision_config remains effective after merge.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "thinker_text_config",
        "vision_config": {"source": "top"},
        "thinker_config": {
            "text_config": {
                "hidden_size": 1024,
                "num_attention_heads": 8,
            },
            "vision_config": {"source": "thinker"},
        },
    }))

    assert cfg.hidden_size == 1024
    assert cfg.raw["vision_config"]["source"] == "top"


def test_nonstandard_key_fallbacks_cover_n_embed_num_heads_n_layers_hidden_dim_and_norm_eps():
    """Intent: cover fallback keys not exercised by existing tests.
    Preconditions: standard hidden/heads/layers/intermediate/epsilon keys are absent.
    Postconditions: parser uses n_embed/num_heads/n_layers/hidden_dim/norm_eps.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "custom",
        "n_embed": 1536,
        "num_heads": 24,
        "n_layers": 18,
        "hidden_dim": 6144,
        "norm_eps": 1e-4,
    }))

    assert cfg.hidden_size == 1536
    assert cfg.num_attention_heads == 24
    assert cfg.num_hidden_layers == 18
    assert cfg.intermediate_size == 6144
    assert cfg.rms_norm_eps == pytest.approx(1e-4)


def test_num_attention_heads_falls_back_to_n_heads():
    """Intent: exercise n_heads fallback branch for attention head count.
    Preconditions: num_attention_heads and num_heads are absent, n_heads is present.
    Postconditions: num_attention_heads is sourced from n_heads.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "model_type": "dim_encoder",
        "dim": 768,
        "n_heads": 12,
    }))

    assert cfg.hidden_size == 768
    assert cfg.num_attention_heads == 12


def test_zero_token_ids_are_coerced_to_minus_one():
    """Intent: document token-id coercion behavior in dataclass construction.
    Preconditions: config explicitly sets bos/eos/pad token ids to 0.
    Postconditions: resulting ids are -1 because of `or -1` fallback logic.
    """
    cfg = ModelConfig.from_json(json.dumps({
        "hidden_size": 256,
        "num_attention_heads": 4,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }))

    assert cfg.bos_token_id == -1
    assert cfg.eos_token_id == -1
    assert cfg.pad_token_id == -1


def test_create_tiny_provides_expected_defaults():
    """Intent: cover create_tiny default path.
    Preconditions: create_tiny is called with only model_type.
    Postconditions: returned config contains canonical tiny defaults.
    """
    cfg = ModelConfig.create_tiny("unit_test")

    assert cfg.model_type == "unit_test"
    assert cfg.vocab_size == 32
    assert cfg.hidden_size == 16
    assert cfg.intermediate_size == 32
    assert cfg.num_hidden_layers == 2
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 4
    assert cfg.max_position_embeddings == 128


def test_create_tiny_applies_overrides_before_parsing():
    """Intent: cover create_tiny override behavior.
    Preconditions: create_tiny receives user overrides for multiple fields.
    Postconditions: override values appear in the resulting ModelConfig.
    """
    cfg = ModelConfig.create_tiny(
        "unit_test",
        hidden_size=64,
        num_attention_heads=8,
        rope_theta=123456.0,
        hidden_act="silu",
    )

    assert cfg.hidden_size == 64
    assert cfg.num_attention_heads == 8
    assert cfg.rope_theta == pytest.approx(123456.0)
    assert cfg.hidden_act == "silu"
