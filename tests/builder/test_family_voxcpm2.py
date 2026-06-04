"""VoxCPM2 family registration tests."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

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
    for filename in (
        "tokenization_voxcpm2.py",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
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
    assert sources["locenc"].state_dict_prefixes == ("feat_encoder.", "enc_to_lm_proj.")
    assert sources["locenc"].asset_files == (
        "tokenization_voxcpm2.py",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    assert sources["tslm"].config_values["lm_config"]["hidden_size"] == 2048
    assert sources["tslm"].config_values["max_length"] == 8192
    assert sources["tslm"].state_dict_prefixes == (
        "base_lm.",
        "fsq_layer.",
        "stop_proj.",
        "stop_head.",
    )
    assert sources["tslm"].asset_files == sources["locenc"].asset_files
    assert sources["ralm"].config_values["residual_lm_num_layers"] == 8
    assert sources["ralm"].config_values["scalar_quantization_latent_dim"] == 512
    assert sources["ralm"].state_dict_prefixes == (
        "fusion_concat_proj.",
        "residual_lm.",
        "res_to_dit_proj.",
    )
    assert sources["ralm"].asset_files == ()
    assert sources["locdit"].config_keys == (
        "dit_config",
        "dit_config.cfm_config",
        "patch_size",
        "feat_dim",
    )
    assert sources["locdit"].config_values["dit_config"]["hidden_dim"] == 1024
    assert sources["locdit"].config_values["dit_config.cfm_config"]["solver"] == "euler"
    assert sources["locdit"].state_dict_prefixes == ("lm_to_dit_proj.", "feat_decoder.")
    assert sources["locdit"].asset_files == ()
    assert sources["audiovae"].config_values["audio_vae_config"]["out_sample_rate"] == 48000
    assert sources["audiovae"].weight_files == ("audiovae.pth",)
    assert sources["audiovae"].state_dict_prefixes == ()
    assert sources["audiovae"].asset_files == ()


def test_voxcpm2_raw_checkpoint_reports_native_builder_gap(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as error:
        plugin.build_engine(cfg, weights, max_cache_length=16)

    message = str(error.value)
    assert "raw checkpoint sources are present for locenc, tslm, ralm, locdit, audiovae" in message
    assert "native TRT builders are incomplete" in message
    assert "assets: tokenization_voxcpm2.py, tokenizer_config.json" in message
    assert "locdit(config: dit_config, dit_config.cfm_config, patch_size, feat_dim" in message
    assert "audiovae(config: audio_vae_config" in message
    assert "component 'locenc' is not implemented yet" in message
    assert "input binding 'text_utf8'" in message
    assert "output binding 'local_text_features'" in message
    assert "native text-to-audio runtime that writes the TRT WAV artifact" in message


def test_voxcpm2_raw_checkpoint_invokes_native_component_builders(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    calls = []
    path_calls = []

    def make_builder(component_name):
        def _builder(ctx):
            calls.append(
                (
                    ctx.spec.name,
                    ctx.spec.engine_section,
                    ctx.spec.input_artifact,
                    ctx.spec.output_artifact,
                    ctx.model_dir,
                    ctx.precision,
                    ctx.verbose,
                    ctx.source.config_keys,
                    ctx.source.asset_files,
                )
            )
            path_calls.append((ctx.spec.name, ctx.weight_paths, ctx.asset_paths))
            return f"{component_name}-plan".encode("ascii")

        return _builder

    fake_builders = {
        spec.name: make_builder(spec.name)
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=16,
        precision="bf16",
        verbose=True,
    )

    assert sections == {
        spec.engine_section: f"{spec.name}-plan".encode("ascii")
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    assert calls == [
        (
            "locenc",
            "locenc_engine_plan",
            "text_utf8",
            "local_text_features",
            tmp_path,
            "bf16",
            True,
            ("encoder_config", "patch_size", "feat_dim"),
            (
                "tokenization_voxcpm2.py",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
            ),
        ),
        (
            "tslm",
            "tslm_engine_plan",
            "local_text_features",
            "semantic_lm_states",
            tmp_path,
            "bf16",
            True,
            ("lm_config", "max_length"),
            (
                "tokenization_voxcpm2.py",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
            ),
        ),
        (
            "ralm",
            "ralm_engine_plan",
            "semantic_lm_states",
            "acoustic_residual_states",
            tmp_path,
            "bf16",
            True,
            (
                "lm_config",
                "residual_lm_num_layers",
                "scalar_quantization_latent_dim",
                "scalar_quantization_scale",
            ),
            (),
        ),
        (
            "locdit",
            "locdit_engine_plan",
            "acoustic_residual_states",
            "audio_vae_latents",
            tmp_path,
            "bf16",
            True,
            ("dit_config", "dit_config.cfm_config", "patch_size", "feat_dim"),
            (),
        ),
        (
            "audiovae",
            "audiovae_engine_plan",
            "audio_vae_latents",
            "waveform_f32",
            tmp_path,
            "bf16",
            True,
            (
                "audio_vae_config",
                "audio_vae_config.sample_rate",
                "audio_vae_config.out_sample_rate",
            ),
            (),
        ),
    ]
    tokenizer_asset_paths = (
        tmp_path / "tokenization_voxcpm2.py",
        tmp_path / "tokenizer_config.json",
        tmp_path / "tokenizer.json",
        tmp_path / "special_tokens_map.json",
    )
    assert path_calls == [
        ("locenc", (tmp_path / "model.safetensors",), tokenizer_asset_paths),
        ("tslm", (tmp_path / "model.safetensors",), tokenizer_asset_paths),
        ("ralm", (tmp_path / "model.safetensors",), ()),
        ("locdit", (tmp_path / "model.safetensors",), ()),
        ("audiovae", (tmp_path / "audiovae.pth",), ()),
    ]


def test_voxcpm2_component_context_loads_checkpoint_inputs(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {"locenc.weight": np.array([1.0, 2.0], dtype=np.float32)},
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]
    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    locenc_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["locenc"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["locenc"],
        precision="bf16",
        verbose=False,
    )
    np.testing.assert_array_equal(
        locenc_ctx.load_safetensor("locenc.weight"),
        np.array([1.0, 2.0], dtype=np.float32),
    )

    def _fake_torch_load(path, *, map_location, weights_only):
        assert path == tmp_path / "audiovae.pth"
        assert map_location == "cpu"
        assert weights_only is True
        return {"state_dict": {"decoder.weight": np.array([3.0], dtype=np.float32)}}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=_fake_torch_load))
    audiovae_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["audiovae"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["audiovae"],
        precision="bf16",
        verbose=False,
    )

    state_dict = audiovae_ctx.load_torch_checkpoint()
    np.testing.assert_array_equal(
        state_dict["decoder.weight"],
        np.array([3.0], dtype=np.float32),
    )


def test_voxcpm2_component_context_loads_stage_scoped_safetensors(tmp_path):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {
            "base_lm.layers.0.weight": np.array([1.0], dtype=np.float32),
            "enc_to_lm_proj.weight": np.array([2.0], dtype=np.float32),
            "feat_decoder.estimator.weight": np.array([3.0], dtype=np.float32),
            "feat_encoder.layers.0.weight": np.array([4.0], dtype=np.float32),
            "residual_lm.layers.0.weight": np.array([5.0], dtype=np.float32),
            "unrelated.weight": np.array([6.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]
    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    locenc_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["locenc"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["locenc"],
        precision="bf16",
        verbose=False,
    )
    locenc_group = locenc_ctx.load_safetensor_group()

    assert tuple(locenc_group) == (
        "enc_to_lm_proj.weight",
        "feat_encoder.layers.0.weight",
    )
    np.testing.assert_array_equal(
        locenc_group["feat_encoder.layers.0.weight"],
        np.array([4.0], dtype=np.float32),
    )

    ralm_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["ralm"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["ralm"],
        precision="bf16",
        verbose=False,
    )
    assert tuple(ralm_ctx.load_safetensor_group()) == (
        "residual_lm.layers.0.weight",
    )

    with pytest.raises(KeyError, match="found no safetensors tensors matching"):
        locenc_ctx.load_safetensor_group(("missing_prefix.",))


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
