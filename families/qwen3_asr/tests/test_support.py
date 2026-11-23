# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default-task policy for Qwen3-ASR."""

from families.qwen3_asr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

_ASR = {
    "model_type": "qwen3_asr",
    "architectures": ["Qwen3ASRForConditionalGeneration"],
    "audio_config": {"model_type": "qwen3_asr_encoder"},
    "text_config": {"model_type": "qwen3"},
}


def test_speech_recognition_checkpoint_selects_transcription() -> None:
    support = describe(ModelMetadata(config=_ASR, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"
    assert support.default_precision == "bf16"


def test_checkpoint_resolves_to_exactly_this_family() -> None:
    family, support = resolve_family(ModelMetadata(config=_ASR, model_index={}))
    assert family == "qwen3_asr"
    assert support.default_task == "speech_transcription"


def test_forced_aligner_head_is_not_claimed() -> None:
    # Same model_type, a timestamp-classification head this family does not build.
    aligner = {**_ASR, "architectures": ["Qwen3ASRForTokenClassification"]}
    assert describe(ModelMetadata(config=aligner, model_index={})) is None


def test_original_release_layout_is_not_claimed() -> None:
    # Qwen/Qwen3-ASR-0.6B (without -hf) and Qwen/Qwen3-ForcedAligner-0.6B declare the same
    # model_type and architecture but keep their configuration under thinker_config.
    original = {
        "model_type": "qwen3_asr",
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "thinker_config": {"audio_config": {}, "text_config": {}},
    }
    assert describe(ModelMetadata(config=original, model_index={})) is None
    assert describe(ModelMetadata(config={**_ASR, "thinker_config": {}}, model_index={})) is None
    without_audio = {key: value for key, value in _ASR.items() if key != "audio_config"}
    assert describe(ModelMetadata(config=without_audio, model_index={})) is None


def test_neighbouring_audio_models_are_not_claimed() -> None:
    for model_type in ("qwen3_asr_encoder", "qwen3", "qwen3_omni_moe", "glmasr", "whisper"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
    assert describe(ModelMetadata(config={}, model_index={})) is None
