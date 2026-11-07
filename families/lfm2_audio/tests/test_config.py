# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.lfm2_audio.config import validate_dense_lfm2_config
from families.lfm2_audio.support import describe
from tensorrt_model_connect.model_support import ModelMetadata

FIXTURES = Path(__file__).with_name("fixtures")


def test_checkpoint_identity_and_tasks():
    raw = json.loads((FIXTURES / "config.json").read_text())
    support = describe(ModelMetadata(raw, {}))
    assert support is not None
    assert support.default_task == "offline_speech_dialogue"
    assert set(support.tasks) == {
        "offline_speech_dialogue",
        "speech_transcription",
        "speech_to_speech_response",
        "text_to_speech",
        "streaming_text_to_speech",
        "audio_text_to_text",
        "text_conversation",
    }
    assert support.default_precision == "bf16"
    assert describe(ModelMetadata(raw["lfm"], {})) is None
    assert (
        describe(ModelMetadata({"architectures": ["Lfm2VlForConditionalGeneration"]}, {})) is None
    )


def test_backbone_geometry_matches_checkpoint():
    raw = json.loads((FIXTURES / "config.json").read_text())["lfm"]
    cfg = validate_dense_lfm2_config(SimpleNamespace(raw=raw))
    assert (
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_conv_layers,
        cfg.num_attention_layers,
    ) == (2048, 8192, 10, 6)
    assert (cfg.head_dim, cfg.conv_l_cache, cfg.default_cache_length) == (64, 3, 32768)


def test_detokenizer_is_distinct_sliding_lfm():
    raw = json.loads((FIXTURES / "detokenizer_config.json").read_text())
    assert raw["sliding_window"] == 30
    raw["architectures"] = ["Lfm2ForCausalLM"]
    raw["layer_types"] = [
        "full_attention" if x == "sliding_attention" else x for x in raw["layer_types"]
    ]
    cfg = validate_dense_lfm2_config(SimpleNamespace(raw=raw))
    assert (
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_conv_layers,
        cfg.num_attention_layers,
    ) == (512, 2304, 5, 3)


@pytest.mark.parametrize(
    "field,value", [("num_key_value_heads", 7), ("conv_dim", 1024), ("conv_L_cache", 0)]
)
def test_rejects_incompatible_backbone(field, value):
    raw = json.loads((FIXTURES / "config.json").read_text())["lfm"]
    raw[field] = value
    with pytest.raises(ValueError):
        validate_dense_lfm2_config(SimpleNamespace(raw=raw))
