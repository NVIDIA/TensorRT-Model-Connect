# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phonon ownership must not claim Pocket TTS or unrelated speech models."""

from families.phonon.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_exact_identity_and_default():
    metadata = ModelMetadata(config={"model_type": "gradium_phonon_tts"}, model_index={})
    family, support = resolve_family(metadata)
    assert family == "phonon"
    assert support.tasks == ("text_to_speech",)
    assert support.default_task == "text_to_speech"
    assert support.default_precision == "fp32"


def test_other_speech_models_are_not_owned():
    for model_type in ("pocket_tts", "bark", "gradium_phonon_tts_future", "whisper"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_architecture_identity():
    metadata = ModelMetadata(
        config={"architectures": ["GradiumPhononTTSForConditionalGeneration"]}, model_index={}
    )
    assert describe(metadata).default_task == "text_to_speech"
