# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from families.cohere_asr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config",
    [{"model_type": "cohere_asr"}, {"architectures": ["CohereAsrForConditionalGeneration"]}],
)
def test_exact_identity_and_default(config):
    metadata = ModelMetadata(config=config, model_index={})
    support = describe(metadata)
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"
    assert resolve_family(metadata)[0] == "cohere_asr"


@pytest.mark.parametrize("kind", ["cohere", "canary", "whisper", "parakeet", "cohere_asr_v2", ""])
def test_other_architectures_not_claimed(kind):
    assert describe(ModelMetadata(config={"model_type": kind}, model_index={})) is None
