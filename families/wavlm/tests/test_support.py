# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for WavLM."""

from families.wavlm.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_transcription_is_offered():
    support = describe(ModelMetadata(config={"model_type": "wavlm"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_ctc_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["WavLMForCTC"]}, model_index={}))
    assert support is not None
    assert support.default_task == "speech_transcription"


def test_qualifies_at_fp32():
    # fp32 stays the default: it is the precision that matches the reference
    # transcript exactly. fp16 is built only when asked for (precision=fp16).
    support = describe(ModelMetadata(config={"model_type": "wavlm"}, model_index={}))
    assert support.default_precision == "fp32"


def test_the_sibling_encoders_are_not_claimed():
    # wav2vec2, HuBERT and UniSpeech share WavLM's convolutional front end and
    # most of its tensor names, but none has the gated relative position bias,
    # so none of them is built by this family.
    for model_type in ("wav2vec2", "hubert", "unispeech", "unispeech-sat", "wav2vec2-conformer"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_the_other_wavlm_heads_offer_only_transcription():
    # The model_type is shared by every WavLM head. The pretrained encoder
    # (microsoft/wavlm-large) and the x-vector or classification heads resolve
    # here and are refused by the builder by name, because none has a CTC head.
    for architecture in ("WavLMModel", "WavLMForXVector", "WavLMForSequenceClassification"):
        support = describe(ModelMetadata(
            config={"model_type": "wavlm", "architectures": [architecture]}, model_index={}))
        assert support is not None
        assert support.tasks == ("speech_transcription",)
