"""Unit tests for the VoxCPM2 family support layer.

Trace: ARCH-FAM-001, UD-FAM-VOXCPM2
Intent: Validate VoxCPM2 detection and explicit runtime-support blocker.
Preconditions: Synthetic VoxCPM2 config is available on disk.
Postconditions: Plugin extracts architecture metadata and refuses unsupported TRT builds.
"""

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.config import ModelConfig
import tensorrt_model_connect.families.voxcpm2 as voxcpm2_mod


def _cfg() -> ModelConfig:
    return ModelConfig.from_json(json.dumps(_raw_config()))


def _raw_config() -> dict:
    return {
        "architecture": "voxcpm2",
        "lm_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 73448,
        },
        "residual_lm_num_layers": 8,
        "encoder_config": {
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 12,
        },
        "dit_config": {
            "hidden_dim": 1024,
            "num_heads": 16,
            "num_layers": 12,
        },
        "audio_vae_config": {
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        },
        "dtype": "bfloat16",
        "max_length": 8192,
    }


def test_matches_aliases_and_rejects_other_tts_families() -> None:
    plugin = voxcpm2_mod.plugin
    assert plugin.matches("voxcpm2")
    assert plugin.matches("vox-cpm2")
    assert plugin.matches("openbmb/VoxCPM2")
    assert not plugin.matches("bark")
    assert not plugin.matches("magpie_tts")


def test_load_weights_extracts_metadata_and_build_engine_blocks(tmp_path) -> None:
    model_dir = tmp_path / "voxcpm2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_raw_config()))

    cfg = _cfg()
    weights = voxcpm2_mod.plugin.load_weights(str(model_dir), cfg)

    assert weights["_model_format"] == "voxcpm2"
    assert weights["_requires_python_package"] == "voxcpm"
    assert cfg.raw["_voxcpm2_architecture"]["lm_hidden_size"] == 2048
    assert cfg.raw["_voxcpm2_architecture"]["locenc_layers"] == 12
    assert cfg.raw["_voxcpm2_architecture"]["locdit_layers"] == 12
    assert cfg.raw["_voxcpm2_architecture"]["audio_vae_out_sample_rate"] == 48000

    with pytest.raises(NotImplementedError, match="text_to_audio_voxcpm2"):
        voxcpm2_mod.plugin.build_engine(cfg, weights, 256)


def test_bundle_config_overrides_document_upstream_runtime() -> None:
    cfg = _cfg()
    cfg.raw["_voxcpm2_architecture"] = {
        "audio_vae_sample_rate": 16000,
        "audio_vae_out_sample_rate": 48000,
        "dtype": "bfloat16",
    }

    overrides = voxcpm2_mod.plugin.get_bundle_config_overrides(cfg)

    assert overrides["runtime_strategy"] == "text_to_audio_voxcpm2"
    assert overrides["audio_task"] == "text_to_audio"
    assert overrides["voxcpm2"]["upstream_library"] == "voxcpm"
    assert overrides["voxcpm2"]["sample_rate"] == 48000
    assert overrides["voxcpm2"]["reference_sample_rate"] == 16000
    assert "LocEnc" in overrides["voxcpm2"]["architecture"]
