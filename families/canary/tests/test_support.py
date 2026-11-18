# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity tests for Canary."""

import importlib
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "canary"},
        {"model_type": "canary_asr"},
        {"model_type": "enc_dec_multi_task"},
    ],
)
def test_canary_discovery(config):
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "canary"
    assert support.tasks == ("speech_transcription", "speech_translation")
    assert support.default_task == "speech_transcription"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "whisper"},
        {"model_type": "wav2vec2"},
        {"model_type": "parakeet_tdt"},
        {},
    ],
)
def test_canary_does_not_claim_other_families(config):
    describe = importlib.import_module("families.canary.support").describe
    assert describe(ModelMetadata(config, {}, ("canary.nemo",))) is None
