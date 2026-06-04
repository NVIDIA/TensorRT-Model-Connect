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


def _write_raw_voxcpm2_checkpoint(tmp_path) -> ModelConfig:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "lm_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 73448,
                },
                "patch_size": 4,
                "feat_dim": 64,
                "scalar_quantization_latent_dim": 512,
                "scalar_quantization_scale": 9,
                "residual_lm_num_layers": 8,
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
                    "cfm_config": {
                        "solver": "euler",
                        "inference_cfg_rate": 2.0,
                    },
                },
                "audio_vae_config": {
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                    "latent_dim": 64,
                },
                "max_length": 8192,
                "sample_rate": 48000,
                "cfg_value": 2.0,
                "inference_timesteps": 10,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"raw-safe-tensors")
    (tmp_path / "audiovae.pth").write_bytes(b"raw-audio-vae")
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
    assert weights["_voxcpm2_raw_component_sources"] == {}
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


def test_voxcpm2_raw_checkpoint_sources_are_recorded(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]

    assert tuple(sources) == ("locenc", "tslm", "ralm", "locdit", "audiovae")
    assert sources["locenc"].config_keys == ("encoder_config", "patch_size", "feat_dim")
    assert sources["locenc"].config_values["encoder_config"]["hidden_dim"] == 1024
    assert sources["locenc"].config_values["patch_size"] == 4
    assert sources["locenc"].config_values["feat_dim"] == 64
    assert sources["locenc"].weight_files == ("model.safetensors",)
    assert sources["tslm"].config_values["lm_config"]["hidden_size"] == 2048
    assert sources["tslm"].config_values["max_length"] == 8192
    assert sources["ralm"].config_values["residual_lm_num_layers"] == 8
    assert sources["ralm"].config_values["scalar_quantization_latent_dim"] == 512
    assert sources["locdit"].config_keys == (
        "dit_config",
        "dit_config.cfm_config",
        "patch_size",
        "feat_dim",
    )
    assert sources["locdit"].config_values["dit_config"]["hidden_dim"] == 1024
    assert sources["locdit"].config_values["dit_config.cfm_config"]["solver"] == "euler"
    assert sources["audiovae"].config_values["audio_vae_config"]["out_sample_rate"] == 48000
    assert sources["audiovae"].weight_files == ("audiovae.pth",)


def test_voxcpm2_raw_checkpoint_reports_native_builder_gap(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as error:
        plugin.build_engine(cfg, weights, max_cache_length=16)

    message = str(error.value)
    assert "raw checkpoint sources are present for locenc, tslm, ralm, locdit, audiovae" in message
    assert "native TRT builders are not implemented yet" in message
    assert "locdit(config: dit_config, dit_config.cfm_config, patch_size, feat_dim" in message
    assert "audiovae(config: audio_vae_config" in message
    assert "native text-to-audio runtime that writes the TRT WAV artifact" in message


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
