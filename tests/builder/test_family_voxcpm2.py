"""VoxCPM2 family registration tests."""

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.runtime_config import clear_for_testing, lookup
from tensorrt_model_connect.runtime_config.schemas import load_all


def _voxcpm2_config(tmp_path) -> ModelConfig:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "sample_rate": 48000,
                "cfg_value": 2.0,
                "inference_timesteps": 10,
            }
        ),
        encoding="utf-8",
    )
    return ModelConfig.from_dir(tmp_path)


def test_model_config_uses_architecture_fallback_for_voxcpm2(tmp_path):
    cfg = _voxcpm2_config(tmp_path)

    assert cfg.model_type == "voxcpm2"
    assert cfg.architectures == ["voxcpm2"]


def test_voxcpm2_plugin_records_metadata_and_audio_defaults(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)

    assert plugin.matches(cfg.model_type)
    weights = plugin.load_weights(str(tmp_path), cfg)
    assert weights["_architecture"] == "voxcpm2"
    assert weights["_voxcpm2_components"] == ("locenc", "tslm", "ralm", "locdit")
    assert weights["_sample_rate"] == 48000
    assert weights["_cfg_value"] == 2.0
    assert weights["_inference_timesteps"] == 10

    audio_cfg = plugin.get_audio_config(cfg)
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["reference_sample_rate"] == 16000
    assert audio_cfg["voxcpm2_cfg_value"] == 2.0
    assert audio_cfg["voxcpm2_inference_timesteps"] == 10


def test_voxcpm2_build_boundary_is_explicit(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="LocEnc, TSLM, RALM, and LocDiT"):
        plugin.build_engine(cfg, weights, max_cache_length=16)


def test_audio_voxcpm2_schema_defaults_and_validators():
    clear_for_testing()
    try:
        load_all()
        schema = lookup("audio_voxcpm2")
        assert schema is not None
        fields = {field.name: field for field in schema.fields}

        assert fields["cfg_value"].default == 2.0
        assert fields["inference_timesteps"].default == 10
        assert fields["normalize"].default is True
        assert fields["denoise"].default is True
        assert fields["retry_badcase"].default is True
        assert fields["retry_badcase_max_times"].default == 3
        assert fields["retry_badcase_ratio_threshold"].default == 6.0
        assert fields["seed"].default == -1
        assert fields["cfg_value"].validator(0.0)
        assert not fields["cfg_value"].validator(-0.1)
        assert fields["inference_timesteps"].validator(10)
        assert not fields["inference_timesteps"].validator(True)
    finally:
        clear_for_testing()
