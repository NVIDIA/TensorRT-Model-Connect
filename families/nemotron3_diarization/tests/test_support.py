# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for Nemotron 3 Diarization."""

from families.nemotron3_diarization.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_speaker_diarization_is_offered():
    support = describe(ModelMetadata(config={"model_type": "nemotron3_diarization"}, model_index={}))
    assert support is not None
    assert support.tasks == ("audio_to_timed_labels",)
    assert support.default_task == "audio_to_timed_labels"
    assert support.default_precision == "fp32"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(
            config={"architectures": ["Nemotron3DiarizationForAudioFrameClassification"]},
            model_index={},
        )
    )
    assert support is not None
    assert support.default_task == "audio_to_timed_labels"


def test_other_speech_models_are_not_claimed():
    # Sortformer's sibling checkpoints and the streaming ASR encoder share the
    # NeMo lineage, not this head, chunking and speaker cache.
    for model_type in ("nemotron_asr_streaming", "nemotron3_5_transcribe", "parakeet", "wav2vec2"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
