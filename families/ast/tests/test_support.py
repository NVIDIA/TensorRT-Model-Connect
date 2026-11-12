# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for AST."""

from families.ast.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_audio_tagging_is_offered():
    support = describe(
        ModelMetadata(config={"model_type": "audio-spectrogram-transformer"}, model_index={})
    )
    assert support is not None
    assert support.tasks == ("audio_to_class_scores",)
    assert support.default_task == "audio_to_class_scores"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["ASTForAudioClassification"]}, model_index={})
    )
    assert support is not None
    assert support.default_task == "audio_to_class_scores"


def test_other_audio_encoders_are_not_claimed():
    # Wav2Vec2 and Whisper also classify audio, but neither is a patch
    # transformer over a Kaldi filter bank, so neither shares this frontend.
    for model_type in ("wav2vec2", "whisper", "hubert"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
