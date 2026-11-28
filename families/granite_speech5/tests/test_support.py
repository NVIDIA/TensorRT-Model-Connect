# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and Task discovery for Granite Speech 5 CTC."""

from families.granite_speech5.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_model_type_offers_speech_transcription():
    support = describe(ModelMetadata(config={"model_type": "granite_speech5_ctc"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["GraniteSpeech5ForCTC"]}, model_index={}))
    assert support is not None
    assert support.default_task == "speech_transcription"


def test_default_precision_is_the_checkpoint_precision():
    # The checkpoint stores BF16; fp32 is built only when asked for.
    support = describe(ModelMetadata(config={"model_type": "granite_speech5_ctc"}, model_index={}))
    assert support.default_precision == "bf16"


def test_neighbouring_models_are_not_claimed():
    # granite_speech is the Granite Speech 4 encoder-decoder family; the bare encoder has no CTC head.
    for model_type in ("granite_speech", "granite_speech5_encoder", "granite_speech_plus", "granite"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
    for architecture in ("GraniteSpeechForConditionalGeneration", "GraniteSpeech5Encoder", "Wav2Vec2ForCTC"):
        assert describe(ModelMetadata(config={"architectures": [architecture]}, model_index={})) is None
