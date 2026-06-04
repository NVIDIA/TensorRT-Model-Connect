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
    assert plugin.runtime_strategy == "text_to_audio_voxcpm2"
    weights = plugin.load_weights(str(tmp_path), cfg)
    assert weights["_architecture"] == "voxcpm2"
    assert weights["_voxcpm2_components"] == (
        "locenc",
        "tslm",
        "ralm",
        "locdit",
        "audiovae",
    )
    assert weights["_voxcpm2_engine_sections"] == (
        "locenc_engine_plan",
        "tslm_engine_plan",
        "ralm_engine_plan",
        "locdit_engine_plan",
        "audiovae_engine_plan",
    )
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

    with pytest.raises(
        NotImplementedError, match="missing artifacts for locenc, tslm, ralm, locdit, audiovae"
    ):
        plugin.build_engine(cfg, weights, max_cache_length=16)


def test_voxcpm2_build_engine_packages_prebuilt_component_plans(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    plans = {
        "locenc.plan": b"LOCENC",
        "tslm.engine": b"TSLM",
        "ralm.trtplan": b"RALM",
        "locdit_engine_plan.plan": b"LOCDIT",
        "audiovae_engine_plan": b"AUDIOVAE",
    }
    for filename, data in plans.items():
        (tmp_path / filename).write_bytes(data)

    sections = plugin.build_engine(cfg, weights, max_cache_length=16)

    assert sections == {
        "locenc_engine_plan": b"LOCENC",
        "tslm_engine_plan": b"TSLM",
        "ralm_engine_plan": b"RALM",
        "locdit_engine_plan": b"LOCDIT",
        "audiovae_engine_plan": b"AUDIOVAE",
    }


def test_voxcpm2_build_engine_reports_partial_prebuilt_plan_set(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    (tmp_path / "locenc.plan").write_bytes(b"LOCENC")
    (tmp_path / "tslm.plan").write_bytes(b"TSLM")

    with pytest.raises(NotImplementedError) as error:
        plugin.build_engine(cfg, weights, max_cache_length=16)

    message = str(error.value)
    assert "missing artifacts for ralm, locdit, audiovae" in message
    assert "ralm.plan" in message
    assert "audiovae.trtplan" in message


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
