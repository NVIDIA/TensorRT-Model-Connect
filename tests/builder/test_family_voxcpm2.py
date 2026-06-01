"""Tests for VoxCPM2 detection-only family support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_plugin
from tests.e2e_harness.manifest_loader import load_manifest


def _voxcpm2_config() -> ModelConfig:
    return ModelConfig.from_json(json.dumps({
        "architecture": "voxcpm2",
        "lm_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "vocab_size": 73448,
        },
        "dit_config": {
            "hidden_dim": 1024,
            "num_layers": 12,
        },
        "audio_vae_config": {
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        },
        "max_length": 8192,
        "dtype": "bfloat16",
    }))


def test_voxcpm2_plugin_matches_config_model_type():
    plugin = find_plugin("voxcpm2")
    assert plugin is not None
    assert plugin.name == "voxcpm2"
    assert plugin.matches("voxcpm2")
    assert plugin.matches("vox-cpm2")
    assert not plugin.matches("bark")


def test_voxcpm2_audio_config_records_upstream_shape():
    plugin = find_plugin("voxcpm2")
    cfg = _voxcpm2_config()

    audio_cfg = plugin.get_audio_config(cfg)

    assert audio_cfg["voxcpm2"] is True
    assert audio_cfg["voxcpm2_runtime_supported"] is False
    assert audio_cfg["voxcpm2_architecture"] == "voxcpm2"
    assert audio_cfg["voxcpm2_reference_sample_rate"] == 16000
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["voxcpm2_lm_hidden_size"] == 2048
    assert audio_cfg["voxcpm2_lm_layers"] == 28
    assert audio_cfg["voxcpm2_dit_hidden_dim"] == 1024
    assert audio_cfg["voxcpm2_dit_layers"] == 12


def test_voxcpm2_build_fails_before_misrouting_to_bark_or_magpie(tmp_path):
    plugin = find_plugin("voxcpm2")
    cfg = _voxcpm2_config()

    with pytest.raises(NotImplementedError, match="LocEnc"):
        plugin.load_weights(str(tmp_path), cfg, precision="bf16")


def test_voxcpm2_manifest_routes_to_text_to_audio_and_documents_skip():
    manifest = Path(__file__).resolve().parents[1] / "e2e" / "models" / "voxcpm2.json"

    with pytest.warns(UserWarning, match="unknown runtime_strategy"):
        case = load_manifest(manifest)

    assert case.runtime_strategy == "text_to_audio_voxcpm2"
    assert case.task_strategy == "text_to_audio"
    assert case.reference_backend == "custom_python"
    assert "skip_reason" in case.metadata
    assert "voxcpm-based reference runner" in case.metadata["reference_note"]
