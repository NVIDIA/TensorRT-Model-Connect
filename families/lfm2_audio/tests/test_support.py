# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checkpoint ownership and semantic Task declarations."""

from families.lfm2_audio.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_checkpoint_has_one_owner_and_speech_dialogue_default():
    metadata = ModelMetadata({"architectures": ["Lfm2AudioForConditionalGeneration"]}, {})
    family, support = resolve_family(metadata)
    assert family == "lfm2_audio"
    assert support.default_task == "offline_speech_dialogue"
    assert support.default_precision == "bf16"
    assert set(support.tasks) == {
        "offline_speech_dialogue",
        "speech_transcription",
        "speech_to_speech_response",
        "text_to_speech",
        "streaming_text_to_speech",
        "audio_text_to_text",
        "text_conversation",
    }


def test_text_and_vision_checkpoints_do_not_select_audio_owner():
    assert describe(ModelMetadata({"architectures": ["Lfm2ForCausalLM"]}, {})) is None
    assert (
        describe(ModelMetadata({"architectures": ["Lfm2VlForConditionalGeneration"]}, {})) is None
    )
