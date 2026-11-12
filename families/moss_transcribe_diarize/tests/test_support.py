# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity for the public CPU gate."""

import pytest

from families.moss_transcribe_diarize.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "moss_transcribe_diarize"},
        {"architectures": ["MossTranscribeDiarizeForConditionalGeneration"]},
    ],
)
def test_discovery_has_one_owner(config):
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "moss_transcribe_diarize"
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"


def test_builds_at_the_checkpoint_precision_by_default():
    # The checkpoint stores BF16; FP32 is built only when asked for.
    support = describe(ModelMetadata({"model_type": "moss_transcribe_diarize"}, {}))
    assert support is not None
    assert support.default_task == "speech_transcription"
    assert support.default_precision == "bf16"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3"},
        {"model_type": "whisper"},
        {"model_type": "moss_transcribe_diarize_audio"},
        {"architectures": ["MossTranscribeDiarizeModel"]},
        {},
    ],
)
def test_components_and_other_heads_are_not_claimed(config):
    # The Qwen3 text model and the Whisper audio tower are components, and the
    # bare backbone head produces no transcript.
    assert describe(ModelMetadata(config, {})) is None
