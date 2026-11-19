# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for hubert."""

from families.hubert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _describe(**config):
    return describe(ModelMetadata(config=config, model_index={}))


def test_the_ctc_head_offers_transcription():
    support = describe(
        ModelMetadata(config={"model_type": "hubert", "architectures": ["HubertForCTC"]},
                      model_index={})
    )
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_qualifies_at_fp32():
    # fp32 is the checkpoint's stored precision and the one whose transcript
    # matches the reference exactly; fp16 is built only when asked for.
    support = _describe(model_type="hubert", architectures=["HubertForCTC"])
    assert support.default_precision == "fp32"


def test_heads_without_a_transcription_output_are_not_claimed():
    # Same model type, no CTC head: the pretrained encoder and the sequence
    # classifier have nothing this family could transcribe with.
    for architecture in ("HubertModel", "HubertForSequenceClassification"):
        assert _describe(model_type="hubert", architectures=[architecture]) is None
    assert _describe(model_type="hubert") is None


def test_the_sibling_ctc_encoders_are_not_claimed():
    # wav2vec2, WavLM and UniSpeech share HuBERT's convolutional front end and
    # CTC head but are other model types with their own owners or encoder terms.
    for model_type, architecture in (("wav2vec2", "Wav2Vec2ForCTC"), ("wavlm", "WavLMForCTC"),
                                     ("unispeech", "UniSpeechForCTC"),
                                     ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC")):
        assert _describe(model_type=model_type, architectures=[architecture]) is None
