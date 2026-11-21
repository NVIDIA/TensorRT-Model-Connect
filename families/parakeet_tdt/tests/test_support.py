# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity for the public CPU gate."""

import importlib
import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "parakeet_tdt"},
        {"architectures": ["ParakeetForTDT"]},
    ],
)
def test_discovery_has_one_owner(config):
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "parakeet_tdt"
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "parakeet_ctc"},
        {"model_type": "parakeet"},
        {"architectures": ["ParakeetForRNNT"]},
        {},
    ],
)
def test_discovery_does_not_claim_other_parakeet_topologies(config):
    describe = importlib.import_module("families.parakeet_tdt.support").describe
    assert describe(ModelMetadata(config, {}, ("parakeet.nemo",))) is None
