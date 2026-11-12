# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity, task and configuration checks for Voxtral."""

import copy

import pytest

from families.voxtral.config import (
    max_audio_chunks,
    parse_config,
    validate_preprocessor,
)
from families.voxtral.support import describe
from tensorrt_model_connect.model_support import ModelMetadata

# Shape of mzbac/voxtral-mini-3b-4bit-mixed (an MLX export of Voxtral-Mini-3B-2507).
CONFIG = {
    "architectures": ["VoxtralForConditionalGeneration"],
    "model_type": "voxtral",
    "audio_token_id": 24,
    "hidden_size": 3072,
    "projector_hidden_act": "gelu",
    "vocab_size": 131072,
    "text_config": {
        "model_type": "llama", "hidden_size": 3072, "intermediate_size": 8192,
        "num_hidden_layers": 30, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 131072, "rms_norm_eps": 1e-05, "rope_theta": 100000000.0,
        "rope_scaling": None, "max_position_embeddings": 131072, "hidden_act": "silu",
        "attention_bias": False, "mlp_bias": False, "sliding_window": None, "pretraining_tp": 1,
    },
    "audio_config": {
        "model_type": "voxtral_encoder", "hidden_size": 1280, "intermediate_size": 5120,
        "num_hidden_layers": 32, "num_attention_heads": 20, "num_key_value_heads": 20,
        "head_dim": 64, "num_mel_bins": 128, "max_source_positions": 1500,
        "activation_function": "gelu", "scale_embedding": False,
    },
    "quantization": {
        "group_size": 64, "bits": 4,
        "audio_tower.conv1": False,
        "audio_tower.layers.0.fc1": {"group_size": 64, "bits": 6},
        "language_model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 4},
        "lm_head": {"group_size": 128, "bits": 6},
    },
}
CONFIG["quantization_config"] = CONFIG["quantization"]
PREPROCESSOR = {
    "chunk_length": 30, "dither": 0.0, "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128, "hop_length": 160, "n_fft": 400, "n_samples": 480000,
    "nb_max_frames": 3000, "padding_side": "right", "padding_value": 0.0,
    "processor_class": "VoxtralProcessor", "return_attention_mask": False, "sampling_rate": 16000,
}


def _support(config):
    return describe(ModelMetadata(config=config, model_index={}))


def test_speech_transcription_is_offered():
    support = describe(ModelMetadata(config={"model_type": "voxtral"}, model_index={}))
    assert support is not None
    assert support.tasks == ("speech_transcription",)
    assert support.default_task == "speech_transcription"
    assert support.default_precision == "fp16"


def test_architecture_also_resolves():
    support = _support({"architectures": ["VoxtralForConditionalGeneration"]})
    assert support is not None and support.default_task == "speech_transcription"


def test_neighbours_are_not_claimed():
    # The streaming Voxtral Realtime model and the plain Whisper / Llama
    # checkpoints have other owners or none; this family must not claim them.
    for model_type in ("voxtral_realtime", "voxtral_encoder", "whisper", "llama", "mistral"):
        assert _support({"model_type": model_type}) is None
    assert _support({"architectures": ["VoxtralRealtimeForConditionalGeneration"]}) is None


def test_checkpoint_geometry():
    config = parse_config(copy.deepcopy(CONFIG))
    assert config.frames_per_token == 4
    assert config.audio_tokens_per_chunk == 375
    assert config.mel_frames_per_chunk == 3000
    assert config.text.num_attention_heads * config.text.head_dim == 4096
    validate_preprocessor(PREPROCESSOR, config)
    # 4096 positions minus a 3 + 5 token frame leave room for ten 30 s chunks.
    assert max_audio_chunks(4096, config, 8) == 10
    with pytest.raises(ValueError, match="cannot hold one 30 s audio chunk"):
        max_audio_chunks(383, config, 8)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "llava", "model_type"),
        (("architectures",), ["VoxtralModel"], "architectures"),
        (("text_config", "model_type"), "mistral", "text_config.model_type"),
        (("text_config", "rope_scaling"), {"rope_type": "yarn", "factor": 4.0}, "rope_scaling"),
        (("text_config", "sliding_window"), 4096, "sliding-window"),
        (("text_config", "attention_bias"), True, "biases"),
        (("text_config", "tie_word_embeddings"), True, "tied output embeddings"),
        (("text_config", "num_key_value_heads"), 7, "multiple of key/value heads"),
        (("audio_config", "activation_function"), "relu", "gelu"),
        (("audio_config", "intermediate_size"), 5000, "whole encoder frames"),
        (("projector_hidden_act", ), "relu", "projector"),
        (("audio_token_id",), 131072, "audio_token_id"),
    ],
)
def test_unsupported_variants_are_refused(path, value, message):
    config = copy.deepcopy(CONFIG)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        parse_config(config)


@pytest.mark.parametrize(
    "entry",
    [{"group_size": 64, "bits": 7}, {"group_size": 60, "bits": 4}, {"group_size": 64},
     {"group_size": 64, "bits": 4, "mode": "mxfp4"}, True],
)
def test_unsupported_quantization_entries_are_refused(entry):
    config = copy.deepcopy(CONFIG)
    config["quantization"]["lm_head"] = entry
    config["quantization_config"] = config["quantization"]
    with pytest.raises(ValueError, match="MLX quantization entry"):
        parse_config(config)


def test_only_mlx_exports_are_accepted():
    config = copy.deepcopy(CONFIG)
    del config["quantization"]
    del config["quantization_config"]
    with pytest.raises(ValueError, match="MLX affine-quantized export"):
        parse_config(config)


@pytest.mark.parametrize(("key", "value"), [("n_fft", 512), ("sampling_rate", 24000),
                                            ("feature_size", 80), ("dither", 0.1)])
def test_other_front_ends_are_refused(key, value):
    config = parse_config(copy.deepcopy(CONFIG))
    preprocessor = dict(PREPROCESSOR, **{key: value})
    with pytest.raises(ValueError):
        validate_preprocessor(preprocessor, config)
