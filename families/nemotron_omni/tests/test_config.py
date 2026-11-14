# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nested architecture validation does not depend on a training framework."""

import copy

import pytest

from families.nemotron_omni.config import OmniConfig


def fixture():
    return {
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
        "llm_config": {
            "model_type": "nemotron_h",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "hybrid_override_pattern": "ME*",
        },
        "vision_config": {"model_type": "radio"},
        "sound_config": {"model_type": "parakeet"},
        "force_image_size": 512,
        "patch_size": 16,
        "vit_hidden_size": 1280,
        "projector_hidden_size": 20480,
        "downsample_ratio": 0.5,
        "ps_version": "v2",
    }


def test_nested_decoder_owns_hybrid_fields():
    source = fixture()
    before = copy.deepcopy(source)
    config = OmniConfig.from_dict(source)
    assert source == before
    assert config.decoder.hidden_size == 16
    assert config.decoder.attention_size == 16
    assert config.decoder.raw["hybrid_override_pattern"] == "ME*"
    assert "llm_config" not in config.decoder.raw
    assert config.vision == source["vision_config"]
    assert config.audio == source["sound_config"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", True),
        ("head_dim", 0),
        ("num_hidden_layers", 1),
        ("num_key_value_heads", 3),
        ("hybrid_override_pattern", "ME?"),
        ("layer_norm_epsilon", float("nan")),
        ("layer_norm_epsilon", float("inf")),
        ("max_position_embeddings", -1),
        ("model_type", "llama"),
    ],
)
def test_invalid_decoder_topology_is_rejected(field, value):
    source = fixture()
    source["llm_config"][field] = value
    with pytest.raises(ValueError):
        OmniConfig.from_dict(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "nemotron_h"),
        ("sound_config", None),
        ("vision_config", None),
        ("downsample_ratio", 0.25),
        ("ps_version", "v1"),
        ("force_image_size", 513),
    ],
)
def test_missing_modality_is_not_a_text_only_substitute(field, value):
    source = fixture()
    source[field] = value
    with pytest.raises(ValueError):
        OmniConfig.from_dict(source)


def test_exact_public_family_identity_and_complete_tasks():
    from tensorrt_model_connect.model_support import ModelMetadata
    from families.nemotron_omni.support import SUPPORTED_TASKS, describe

    support = describe(ModelMetadata(config=fixture(), model_index={}))
    assert support is not None
    assert support.tasks == SUPPORTED_TASKS
    assert support.default_precision == "fp16"
    assert {"audio_text_to_text", "audio_video_text_to_text", "video_text_to_text"}.issubset(
        support.tasks
    )
    assert describe(ModelMetadata(config={"model_type": "nemotron_h"}, model_index={})) is None
