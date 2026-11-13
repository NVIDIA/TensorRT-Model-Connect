# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pytest
from families.musicgen.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def metadata():
    return {
        "model_type": "musicgen",
        "decoder": {
            "hidden_size": 1536,
            "num_hidden_layers": 48,
            "num_codebooks": 4,
            "vocab_size": 2048,
        },
        "text_encoder": {"model_type": "t5", "d_model": 768},
        "audio_encoder": {"model_type": "encodec", "audio_channels": 1, "sampling_rate": 32000},
    }


def test_exact_owner_and_task():
    family, support = resolve_family(ModelMetadata(metadata(), {}))
    assert family == "musicgen"
    assert support.tasks == ("text_to_audio",)
    assert support.default_task == "text_to_audio"
    assert support.default_precision == "fp32"


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("decoder", "hidden_size", 1024),
        ("decoder", "num_hidden_layers", 24),
        ("decoder", "num_codebooks", 8),
        ("audio_encoder", "audio_channels", 2),
        ("audio_encoder", "sampling_rate", 24000),
        ("text_encoder", "model_type", "umt5"),
    ],
)
def test_other_topologies_are_not_claimed(part, key, value):
    config = copy.deepcopy(metadata())
    config[part][key] = value
    assert describe(ModelMetadata(config, {})) is None
    assert describe(ModelMetadata({"model_type": "musicgen_melody"}, {})) is None
