# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and Task discovery for Granite Speech."""

from families.granite_speech.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_model_type_offers_speech_transcription():
    support = describe(ModelMetadata(config={"model_type": "granite_speech"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(
        config={"architectures": ["GraniteSpeechForConditionalGeneration"]}, model_index={}))
    assert support is not None
    assert support.default_task == "speech_transcription"


def test_default_precision_is_the_checkpoint_precision():
    # The checkpoint stores BF16; fp32 is built only when asked for.
    support = describe(ModelMetadata(config={"model_type": "granite_speech"}, model_index={}))
    assert support.default_precision == "bf16"


def test_neighbouring_models_are_not_claimed():
    # granite is the text-only decoder family; granite_speech_plus and the CTC-only
    # granite_speech5 differ in their encoders and heads and are not built here.
    for model_type in ("granite", "granite_speech_plus", "granite_speech5_ctc", "blip_2_qformer"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
    for architecture in ("GraniteForCausalLM", "GraniteSpeech5ForCTC"):
        assert describe(ModelMetadata(config={"architectures": [architecture]}, model_index={})) is None
