# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for vibevoice."""

from families.vibevoice.support import describe
from tensorrt_model_connect.model_support import ModelMetadata

_ORIGINAL = {
    "model_type": "vibevoice",
    "architectures": ["VibeVoiceForConditionalGeneration"],
    "decoder_config": {"model_type": "qwen2"},
    "acoustic_tokenizer_config": {"model_type": "vibevoice_acoustic_tokenizer"},
    "semantic_tokenizer_config": {"model_type": "vibevoice_semantic_tokenizer"},
    "diffusion_head_config": {"model_type": "vibevoice_diffusion_head"},
}


def test_the_original_layout_offers_text_to_speech_in_bf16():
    support = describe(ModelMetadata(config=dict(_ORIGINAL), model_index={}))
    assert support is not None
    assert support.tasks == ("text_to_speech",)
    assert support.default_task == "text_to_speech"
    assert support.default_precision == "bf16"


def test_the_transformers_conversion_is_not_claimed():
    # Same model type and head, but renamed sub-configs and weights
    # (text_config, audio_config, semantic_model_config): not the layout this
    # family reads, so it must not resolve here.
    converted = {
        "model_type": "vibevoice",
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "text_config": {"model_type": "qwen2"},
        "audio_config": {"model_type": "vibevoice_acoustic_tokenizer"},
        "semantic_model_config": {"model_type": "vibevoice_acoustic_tokenizer_encoder"},
        "diffusion_head_config": {"model_type": "vibevoice_diffusion"},
    }
    assert describe(ModelMetadata(config=converted, model_index={})) is None


def test_neighbouring_vibevoice_models_are_not_claimed():
    for model_type in (
        "vibevoice_asr",
        "vibevoice_acoustic_tokenizer",
        "vibevoice_streaming",
        "qwen2",
    ):
        config = dict(_ORIGINAL, model_type=model_type, architectures=[])
        assert describe(ModelMetadata(config=config, model_index={})) is None
