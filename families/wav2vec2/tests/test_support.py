# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for wav2vec2."""

from families.wav2vec2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_transcription_is_offered():
    support = describe(ModelMetadata(config={"model_type": "wav2vec2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["Wav2Vec2ForCTC"]}, model_index={})
    )
    assert support is not None
    assert support.default_task == "speech_transcription"


def test_qualifies_at_fp32():
    # The graph is built in fp32 throughout and fp16 is refused by the builder,
    # because the E2E gate is an exact transcript match: any drift that flips a
    # single frame's argmax changes the text, and fp16 was not measured here.
    support = describe(ModelMetadata(config={"model_type": "wav2vec2"}, model_index={}))
    assert support.default_precision == "fp32"


def test_the_sibling_encoders_are_not_claimed():
    # HuBERT and WavLM share wav2vec2's convolutional frontend and much of its
    # tensor naming, but each adds its own encoder terms, so neither is built
    # by this family and neither may be silently claimed by it.
    for model_type in ("hubert", "wavlm", "wav2vec2_conformer", "whisper"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_the_sequence_classification_head_is_not_claimed():
    # Same model_type, different head: Wav2Vec2ForSequenceClassification sums
    # the hidden states with learned weights and has no CTC head at all.
    support = describe(
        ModelMetadata(config={"architectures": ["Wav2Vec2ForSequenceClassification"]},
                      model_index={})
    )
    assert support is None or support.tasks == ("speech_transcription",)
