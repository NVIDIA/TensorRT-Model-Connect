# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for VibeVoice-ASR."""

from families.vibevoice_asr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _support(config: dict):
    return describe(ModelMetadata(config=config, model_index={}))


def test_the_released_asr_checkpoint_offers_speech_transcription():
    support = describe(
        ModelMetadata(
            config={"model_type": "vibevoice", "architectures": ["VibeVoiceForASRTraining"]},
            model_index={},
        )
    )
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_it_builds_the_native_bf16_checkpoint():
    support = _support({"architectures": ["VibeVoiceForASRTraining"]})
    assert support.default_precision == "bf16"


def test_the_speech_synthesis_checkpoints_are_not_claimed():
    # VibeVoice-1.5B and VibeVoice-Realtime share model_type "vibevoice" but
    # generate speech; only the ASR architecture belongs to this family.
    assert _support({"model_type": "vibevoice"}) is None
    assert (
        _support(
            {"model_type": "vibevoice", "architectures": ["VibeVoiceForConditionalGeneration"]}
        )
        is None
    )
    assert _support({"model_type": "vibevoice_streaming"}) is None


def test_the_transformers_relayout_is_not_claimed():
    # microsoft/VibeVoice-ASR-HF renames and regroups every tensor; this family
    # reads only the original layout.
    assert (
        _support(
            {
                "model_type": "vibevoice_asr",
                "architectures": ["VibeVoiceAsrForConditionalGeneration"],
            }
        )
        is None
    )
