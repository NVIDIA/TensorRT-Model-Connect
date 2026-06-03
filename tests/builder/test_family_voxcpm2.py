"""Focused tests for VoxCPM2 support-layer registration."""

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_plugin
from tensorrt_model_connect.families.voxcpm2 import plugin
from tensorrt_model_connect.runtime_config import clear_for_testing, lookup
from tensorrt_model_connect.runtime_config.schemas import load_all


def _voxcpm2_config() -> dict:
    return {
        "architecture": "voxcpm2",
        "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000,
            "vocab_size": 73448,
        },
        "encoder_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
        },
        "dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
        },
        "audio_vae_config": {
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        },
        "max_length": 8192,
        "dtype": "bfloat16",
    }


def test_model_config_reads_voxcpm2_nested_lm_config():
    cfg = ModelConfig.from_json(json.dumps(_voxcpm2_config()))

    assert cfg.model_type == "voxcpm2"
    assert cfg.hidden_size == 2048
    assert cfg.intermediate_size == 6144
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.vocab_size == 73448
    assert cfg.max_position_embeddings == 32768
    assert cfg.raw["audio_vae_config"]["out_sample_rate"] == 48000


def test_voxcpm2_plugin_discovery_and_runtime_strategy():
    discovered = find_plugin("voxcpm2")

    assert discovered is not None
    assert discovered.name == "voxcpm2"
    assert discovered.runtime_strategy == "text_to_audio_voxcpm2"
    assert plugin.matches("vox-cpm2")


def test_voxcpm2_audio_config_preserves_upstream_audio_contract():
    cfg = ModelConfig.from_json(json.dumps(_voxcpm2_config()))

    audio_cfg = plugin.get_audio_config(cfg)

    assert audio_cfg["voxcpm2"] is True
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["reference_sample_rate"] == 16000
    assert audio_cfg["max_length"] == 8192
    assert audio_cfg["dtype"] == "bfloat16"
    assert audio_cfg["architecture_stages"] == [
        "LocEnc",
        "TSLM",
        "RALM",
        "LocDiT",
        "AudioVAE V2",
    ]


def test_voxcpm2_build_engine_reports_runtime_blocker():
    cfg = ModelConfig.from_json(json.dumps(_voxcpm2_config()))
    weights = plugin.load_weights("", cfg)

    with pytest.raises(NotImplementedError, match="LocEnc -> TSLM -> RALM -> LocDiT"):
        plugin.build_engine(cfg, weights, 8192)


def test_audio_voxcpm2_schema_registered_with_generation_knobs():
    clear_for_testing()
    try:
        load_all()
        schema = lookup("audio_voxcpm2")
        assert schema is not None
        fields = {field.name: field for field in schema.fields}
        assert fields["cfg_value"].default == 2.0
        assert fields["inference_timesteps"].default == 10
        assert fields["retry_badcase_ratio_threshold"].default == 6.0
    finally:
        clear_for_testing()
