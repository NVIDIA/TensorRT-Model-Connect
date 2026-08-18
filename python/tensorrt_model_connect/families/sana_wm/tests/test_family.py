# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SANA-WM family config and plugin behavior."""

from __future__ import annotations

import importlib
import json
import os
import struct

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")

from tensorrt_model_connect.config import ModelConfig
import tensorrt_model_connect.families.sana_wm.model as sana_wm_mod

sana_wm_plugin_mod = importlib.import_module("tensorrt_model_connect.families.sana_wm.model")
_CACHED_STAGE1_TEXT_ENCODER_DIR = sana_wm_plugin_mod._cached_stage1_text_encoder_dir

_SANA_WM_STAGE1_BUILD_ENV_VARS = tuple(
    name for name, _ in sana_wm_plugin_mod._STAGE1_DENOISER_BUILD_ENV_DEFAULTS
)
_SANA_WM_ENV_VARS = (
    "TRTMC_SANA_WM_DOWNLOAD_WEIGHTS",
    "SANA_WM_MODEL_DIR",
    "SANA_WM_NATIVE_PLAN_DIR",
    "SANA_WM_STAGE1_TOKENIZER_DIR",
    "SANA_WM_TOKENIZER_DIR",
    "SANA_WM_TEXT_ENCODER_DIR",
    "SANA_WM_REFINER_TOKENIZER_DIR",
    "SANA_WM_REFINER_TEXT_ENCODER_DIR",
    "TRTMC_SANA_WM_GDN_PLUGIN_LIBRARY",
    "TRTMC_SANA_WM_PATCH_EMBED_PLUGIN_LIBRARY",
    *_SANA_WM_STAGE1_BUILD_ENV_VARS,
    "TRTMC_SANA_WM_QKV_PROJ_PLUGIN",
)


@pytest.fixture(autouse=True)
def _clear_sana_wm_env(monkeypatch) -> None:
    for name in _SANA_WM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_cached_stage1_text_encoder_dir",
        lambda _raw_config: None,
    )


def _sana_yaml() -> str:
    return """
model:
  model: SanaMSVideoCamCtrl_1600M_P1_D20
  image_size: 720
  mixed_precision: bf16
  fp32_attention: true
  linear_head_dim: 112
  qk_norm: true
  cross_norm: true
  pos_embed_type: wan_rope
  camctrl_type: BidirectionalGDNUCPESinglePathLiteLABothTriton
vae:
  vae_type: LTX2VAE_diffusers
  vae_pretrained: hf://Efficient-Large-Model/SANA-WM_bidirectional
  vae_latent_dim: 128
  vae_downsample_rate: 32
  vae_stride: [8, 32, 32]
text_encoder:
  model: gemma-2-2b-it
  model_max_length: 300
  chi_prompt:
    - 'Generate an "Enhanced prompt".'
    - 'User Prompt: '
scheduler:
  predict_flow_v: true
  noise_schedule: linear_flow
  flow_shift: 9.95
  inference_flow_shift: 9.8
  vis_sampler: flow_dpm-solver
"""


def _write_safetensors_header(path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    offset = 0
    header = {}
    dtype_size = {"F32": 4, "BF16": 2}
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= int(dim)
        size = count * dtype_size[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def _write_native_plan_set(
    model_dir,
    *,
    plan_dir=None,
    include_text_encoder: bool = True,
    include_stage1_denoiser: bool = True,
    include_stage1_segments: bool = False,
    include_vae_encoder: bool = True,
    include_refiner_text_encoder: bool = True,
    include_refiner_text_connector: bool = False,
    include_refiner_denoiser: bool = True,
    include_refiner_vae_decoder: bool = True,
) -> dict[str, bytes]:
    engine_dir = plan_dir or model_dir / "trtmc_engines"
    engine_dir.mkdir(parents=True)
    plans = {
        "text_encoder_0_plan": b"text-encoder-plan",
        "denoiser_plan": b"stage1-dit-plan",
        "sana_wm_vae_encoder_plan": b"vae-encoder-plan",
        "vae_decoder_plan": b"stage1-vae-decoder-plan",
        "sana_wm_refiner_text_encoder_plan": b"refiner-text-encoder-plan",
        "sana_wm_refiner_text_connector_plan": b"refiner-text-connector-plan",
        "sana_wm_refiner_denoiser_plan": b"refiner-denoiser-plan",
        "sana_wm_refiner_vae_decoder_plan": b"refiner-vae-decoder-plan",
    }
    if include_stage1_segments:
        plans.update(
            {
                "sana_wm_stage1_denoiser_block0_3_plan": b"stage1-block0-3-plan",
                "sana_wm_stage1_denoiser_block4_7_plan": b"stage1-block4-7-plan",
                "sana_wm_stage1_denoiser_block8_11_plan": b"stage1-block8-11-plan",
                "sana_wm_stage1_denoiser_block12_15_plan": b"stage1-block12-15-plan",
                "sana_wm_stage1_denoiser_block16_final_plan": b"stage1-block16-final-plan",
            }
        )
    if not include_text_encoder:
        plans.pop("text_encoder_0_plan")
    if not include_stage1_denoiser:
        plans.pop("denoiser_plan")
    if not include_vae_encoder:
        plans.pop("sana_wm_vae_encoder_plan")
    if not include_refiner_text_encoder:
        plans.pop("sana_wm_refiner_text_encoder_plan")
    if not include_refiner_text_connector:
        plans.pop("sana_wm_refiner_text_connector_plan")
    if not include_refiner_denoiser:
        plans.pop("sana_wm_refiner_denoiser_plan")
    if not include_refiner_vae_decoder:
        plans.pop("sana_wm_refiner_vae_decoder_plan")
    for section, data in plans.items():
        (engine_dir / f"{section}.plan").write_bytes(data)
    return plans


def _write_tokenizer(model_dir) -> None:
    tokenizer_dir = model_dir / "text_encoder"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text(
        '{"model": {"type": "Unigram"}}',
        encoding="utf-8",
    )
    refiner_tokenizer_dir = model_dir / "refiner" / "text_encoder"
    refiner_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (refiner_tokenizer_dir / "tokenizer.json").write_text(
        '{"model": {"type": "Unigram"}, "refiner": true}',
        encoding="utf-8",
    )


def _write_text_encoder_config(model_dir) -> None:
    text_encoder_dir = model_dir / "text_encoder"
    text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 16,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )


def _write_vae_weights_marker(model_dir) -> None:
    vae_dir = model_dir / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    (vae_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"placeholder")


def _write_refiner_diffusers_markers(model_dir) -> None:
    transformer_dir = model_dir / "refiner" / "transformer"
    connectors_dir = model_dir / "refiner" / "connectors"
    text_encoder_dir = model_dir / "refiner" / "text_encoder"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    connectors_dir.mkdir(parents=True, exist_ok=True)
    text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"refiner-transformer")
    (connectors_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"refiner-connectors")


def test_sana_wm_yaml_config_parses_from_dir(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")

    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "sana_wm"
    assert cfg.hidden_size == 2240
    assert cfg.num_hidden_layers == 20
    assert cfg.num_attention_heads == 20
    assert cfg.head_dim == 112
    assert cfg.raw["patch_size"] == [1, 1, 1]
    assert cfg.max_position_embeddings == 300
    assert cfg.raw["runtime_strategy"] == "diffusion_sana_wm"
    assert cfg.raw["video_num_frames"] == 321
    assert cfg.raw["video_height"] == 704
    assert cfg.raw["video_width"] == 1280
    assert cfg.raw["sana_wm_config"]["vae"]["vae_latent_dim"] == 128


@pytest.mark.parametrize(
    ("model_name", "hidden_size", "num_layers", "num_heads", "patch_size"),
    [
        ("SanaMSVideoCamCtrl_1600M_P2S1_D20", 2240, 20, 20, [1, 1, 1]),
        ("SanaMSVideoCamCtrl_2000M_P2_D20", 2304, 20, 18, [1, 2, 2]),
        ("SanaMSVideoCamCtrl_4800M_P2_D60", 2240, 60, 20, [1, 2, 2]),
    ],
)
def test_sana_wm_yaml_uses_upstream_variant_constructors(
    tmp_path,
    model_name,
    hidden_size,
    num_layers,
    num_heads,
    patch_size,
) -> None:
    (tmp_path / "config.yaml").write_text(
        _sana_yaml().replace("SanaMSVideoCamCtrl_1600M_P1_D20", model_name),
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.hidden_size == hidden_size
    assert cfg.num_hidden_layers == num_layers
    assert cfg.num_attention_heads == num_heads
    assert cfg.raw["patch_size"] == patch_size


def test_sana_wm_plugin_emits_native_runtime_config(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    cfg = ModelConfig.from_dir(tmp_path)

    plugin = sana_wm_mod
    assert plugin.matches("sana_wm")
    assert plugin.matches("SanaMSVideoCamCtrl_1600M_P1_D20")
    assert not plugin.matches("ltx_video")

    weights = plugin.load_weights(str(tmp_path), cfg)
    assert weights["_model_format"] == "sana_wm_yaml"

    overrides = plugin.get_bundle_config_overrides(cfg)
    assert overrides["runtime_strategy"] == "diffusion_sana_wm"
    assert overrides["engine_backend"] == "none"
    assert overrides["sana_wm_hf_id"] == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert overrides["sana_wm_action"] == "w-80,jw-40,w-40,lw-60,w-100"
    assert overrides["sana_wm_translation_speed"] == 0.055
    assert overrides["sana_wm_rotation_speed_deg"] == 1.2
    assert overrides["sana_wm_default_intrinsics"] == pytest.approx(
        [797.87866, 830.0503, 844.2675, 463.7225]
    )
    assert overrides["video_num_frames"] == 321
    assert overrides["fps"] == 16
    assert overrides["num_inference_steps"] == 60
    assert overrides["guidance_scale"] == 5.0
    assert overrides["vae_time_stride"] == 8
    assert overrides["vae_spatial_stride"] == 32
    assert overrides["flow_shift"] == 9.8
    assert overrides["text_encoder_name"] == "gemma-2-2b-it"
    assert overrides["text_encoder_max_length"] == 300
    assert overrides["sana_wm_refiner_text_max_length"] == 1024
    assert overrides["sana_wm_chi_prompt"] == 'Generate an "Enhanced prompt".\nUser Prompt: '


def test_sana_wm_plugin_rejects_build_without_native_plans(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    cfg = ModelConfig.from_dir(tmp_path)

    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "pure C++ builds require a complete native TensorRT component set" in message
    assert "SanaMSVideoCamCtrl DiT" in message
    assert "LTX-2 VAE encoder" in message
    assert "No ONNX fallback is used" in message
    assert "TRTMC_SANA_WM_DOWNLOAD_WEIGHTS=1" in message

    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert overrides["engine_backend"] == "none"
    assert "sana_wm_allow_python_bridge" not in overrides


def test_sana_wm_plugin_reports_native_builder_gap_for_full_snapshot(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_safetensors_header(
        tmp_path / "dit" / "sana_wm_1600m_720p.safetensors",
        {
            "x_embedder.proj.weight": ("F32", [2240, 128, 1, 1, 1]),
            "y_embedder.y_proj.fc1.weight": ("F32", [2240, 2304]),
            "y_embedder.y_embedding": ("BF16", [300, 2304]),
            "final_layer.linear.weight": ("F32", [128, 2240]),
            "plucker_embedder.proj.weight": ("F32", [2240, 48, 1, 1, 1]),
            "raymap_embedder.proj.weight": ("F32", [2240, 3, 1, 1, 1]),
            "blocks.0.attn.qkv.weight": ("F32", [6720, 2240]),
        },
    )
    (tmp_path / "vae").mkdir()
    _write_refiner_diffusers_markers(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "raw component weights that the direct TensorRT builders can consume" in message
    assert "No ONNX fallback is used" in message
    assert "TRTMC_SANA_WM_DOWNLOAD_WEIGHTS" not in message
    assert "stage-1 Gemma text encoder" in message
    assert "Gemma3 refiner text encoder plus LTX-2 text connector stack" in message
    assert "LTX-2 refiner transformer/connectors denoiser" not in message
    assert weights["_refiner_checkpoint"].endswith("refiner")
    assert weights["_refiner_transformer_dir"].endswith("refiner/transformer")
    assert weights["_refiner_connectors_dir"].endswith("refiner/connectors")


def test_sana_wm_plugin_marks_gemma3_refiner_connector_buildable(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_refiner_diffusers_markers(tmp_path)
    (tmp_path / "refiner" / "text_encoder" / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "text_config": {"model_type": "gemma3_text"},
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_dir(tmp_path)

    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_refiner_text_encoder_dir"].endswith("refiner/text_encoder")
    assert weights["_refiner_text_encoder_model_type"] == "gemma3"
    assert weights["_can_build_refiner_text_encoder_plan"] is True
    assert weights["_can_build_refiner_text_connector_plan"] is True
    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "Gemma3 refiner text encoder plus LTX-2 text connector stack" not in message
    assert "LTX-2 refiner transformer/connectors denoiser" not in message


def test_sana_wm_plugin_does_not_treat_gemma3_refiner_as_legacy_buildable(
    tmp_path,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path, include_refiner_text_encoder=False)
    _write_tokenizer(tmp_path)
    refiner_text_encoder_dir = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (refiner_text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "text_config": {"model_type": "gemma3_text"},
            }
        ),
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="sana_wm_refiner_text_encoder_plan"):
        sana_wm_mod.load_weights(str(tmp_path), cfg)


def test_sana_wm_plugin_omits_buildable_text_encoder_from_builder_gap(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_text_encoder_config(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "stage-1 Gemma text encoder" not in message
    assert "SanaMSVideoCamCtrl DiT" in message
    assert "LTX-2 VAE encoder" in message


def test_sana_wm_plugin_downloads_stage1_text_encoder_when_opted_in(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    downloaded = tmp_path / "downloaded-gemma"
    downloaded.mkdir()
    (downloaded / "config.json").write_text('{"model_type": "gemma"}', encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_download(raw_config):
        captured["raw_config"] = raw_config
        return downloaded

    monkeypatch.setenv("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS", "1")
    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_download_stage1_text_encoder_dir",
        fake_download,
    )
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_stage1_text_encoder_dir"] == str(downloaded)
    assert captured["raw_config"] is cfg.raw
    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.build_engine(cfg, weights, 256)
    assert "stage-1 Gemma text encoder" not in str(exc_info.value)


def test_sana_wm_stage1_text_encoder_defaults_to_public_mirror() -> None:
    raw_config = {"text_encoder": {"model": "gemma-2-2b-it"}}

    assert (
        sana_wm_plugin_mod._stage1_text_encoder_hf_id(raw_config)
        == "Efficient-Large-Model/gemma-2-2b-it"
    )


def test_sana_wm_plugin_reuses_cached_stage1_text_encoder(
    tmp_path,
    monkeypatch,
) -> None:
    cached = tmp_path / "cached-gemma"
    (cached / "config.json").parent.mkdir(parents=True)
    (cached / "config.json").write_text(
        '{"model_type": "gemma"}',
        encoding="utf-8",
    )
    _write_safetensors_header(
        cached / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", [8, 4])},
    )
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(cached)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved = _CACHED_STAGE1_TEXT_ENCODER_DIR({"text_encoder": {"model": "gemma-2-2b-it"}})

    assert resolved == cached
    assert captured == {
        "repo_id": "Efficient-Large-Model/gemma-2-2b-it",
        "allow_patterns": list(sana_wm_plugin_mod._STAGE1_TEXT_ENCODER_ALLOW_PATTERNS),
        "local_files_only": True,
    }


def test_sana_wm_gemma3_component_loads_nested_language_model_weights(
    monkeypatch,
) -> None:
    component_plugin = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.components.gemma.model"
    )
    component_config = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.components.gemma.config"
    )
    config = component_config.ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "gemma3",
                "text_config": {
                    "model_type": "gemma3_text",
                    "vocab_size": 8,
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 0,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                },
            }
        )
    )
    captured: dict[str, object] = {}

    def fake_load_standard_weights(model_dir, model_config, **kwargs):
        captured.update(
            {
                "model_dir": model_dir,
                "config": model_config,
                **kwargs,
            }
        )
        return component_plugin.WeightDict(
            {
                "embedding": np.zeros((8, 4), dtype=np.float32),
                "final_norm": np.zeros(4, dtype=np.float32),
            }
        )

    monkeypatch.setattr(
        component_plugin,
        "load_standard_weights",
        fake_load_standard_weights,
    )

    component_plugin.load_weights("/model", config, precision="bf16")

    assert captured["model_dir"] == "/model"
    assert captured["config"] is config
    assert captured["precision"] == "bf16"
    assert captured["model_prefix"] == "language_model.model"
    assert captured["lm_head_key"] == "language_model.lm_head.weight"


def test_sana_wm_stage1_denoiser_builder_uses_direct_trt_api(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    raw_config = {"model_type": "sana_wm"}
    dit_path = tmp_path / "dit" / "sana_wm_1600m_720p.safetensors"

    stage1_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.stage1_dit_builder"
    )

    def fake_load(path, *, precision):
        captured["load_path"] = path
        captured["load_precision"] = precision
        captured["load_env"] = {
            name: os.environ.get(name) for name in _SANA_WM_STAGE1_BUILD_ENV_VARS
        }
        return {"x_embedder.proj.weight": "trt-layout-weight"}

    def fake_build(weights, config, *, precision, verbose):
        captured["build_weights"] = weights
        captured["build_config"] = config
        captured["build_precision"] = precision
        captured["build_verbose"] = verbose
        captured["build_env"] = {
            name: os.environ.get(name) for name in _SANA_WM_STAGE1_BUILD_ENV_VARS
        }
        captured["qkv_plugin_env"] = os.environ.get("TRTMC_SANA_WM_QKV_PROJ_PLUGIN")
        return b"stage1-denoiser-plan"

    monkeypatch.setattr(stage1_builder_mod, "load_sana_wm_stage1_dit_weights", fake_load)
    monkeypatch.setattr(stage1_builder_mod, "build_sana_wm_stage1_dit_engine", fake_build)

    plan = sana_wm_plugin_mod._build_sana_wm_stage1_denoiser_plan(
        dit_path,
        raw_config,
        precision="fp16",
        verbose=True,
    )

    assert plan == b"stage1-denoiser-plan"
    assert captured["load_path"] == dit_path
    assert captured["load_precision"] == "fp16"
    assert captured["build_weights"] == {"x_embedder.proj.weight": "trt-layout-weight"}
    assert captured["build_config"] is raw_config
    assert captured["build_precision"] == "fp16"
    assert captured["build_verbose"] is True
    expected_env = dict(sana_wm_plugin_mod._STAGE1_DENOISER_BUILD_ENV_DEFAULTS)
    assert captured["load_env"] == expected_env
    assert captured["build_env"] == expected_env
    assert captured["qkv_plugin_env"] is None
    for name in _SANA_WM_STAGE1_BUILD_ENV_VARS:
        assert name not in os.environ


def test_sana_wm_refiner_denoiser_builder_uses_direct_trt_api(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    raw_config = {"model_type": "sana_wm", "video_num_frames": 321}
    transformer_dir = tmp_path / "refiner" / "transformer"
    transformer_config = {"num_layers": 2, "attention_head_dim": 128}

    refiner_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.refiner_dit_builder"
    )

    def fake_load(path, *, num_layers, precision):
        captured["load_path"] = path
        captured["load_layers"] = num_layers
        captured["load_precision"] = precision
        return {"proj_in.weight": "trt-layout-weight"}

    def fake_build(weights, config, tx_config, *, precision, verbose):
        captured["build_weights"] = weights
        captured["build_config"] = config
        captured["build_transformer_config"] = tx_config
        captured["build_precision"] = precision
        captured["build_verbose"] = verbose
        return b"refiner-denoiser-plan"

    monkeypatch.setattr(refiner_builder_mod, "load_sana_wm_refiner_dit_weights", fake_load)
    monkeypatch.setattr(refiner_builder_mod, "build_sana_wm_refiner_dit_engine", fake_build)

    plan = sana_wm_plugin_mod._build_sana_wm_refiner_denoiser_plan(
        transformer_dir,
        raw_config,
        transformer_config,
        precision="fp16",
        verbose=True,
    )

    assert plan == b"refiner-denoiser-plan"
    assert captured["load_path"] == transformer_dir
    assert captured["load_layers"] == 2
    assert captured["load_precision"] == "fp16"
    assert captured["build_weights"] == {"proj_in.weight": "trt-layout-weight"}
    assert captured["build_config"] is raw_config
    assert captured["build_transformer_config"] is transformer_config
    assert captured["build_precision"] == "fp16"
    assert captured["build_verbose"] is True


def test_sana_wm_refiner_shape_preserves_timestep_scale_multiplier() -> None:
    refiner_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.refiner_dit_builder"
    )

    shape = refiner_builder_mod.refiner_shape_from_config(
        {"model_type": "sana_wm", "video_num_frames": 321},
        {"timestep_scale_multiplier": 1000, "num_attention_heads": 32, "rope_type": "split"},
    )

    assert shape.timestep_scale_multiplier == 1000.0
    assert shape.rope_type == "split"


def test_sana_wm_refiner_builder_maps_bf16_precision_for_hf_parity() -> None:
    refiner_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.refiner_dit_builder"
    )

    assert refiner_builder_mod._op_np_dtype("bf16") == np.float16
    assert refiner_builder_mod._target_np_dtype("bf16") == np.float32


def test_sana_wm_refiner_text_connector_builder_uses_direct_trt_api(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    raw_config = {"model_type": "sana_wm", "sana_wm_refiner_text_max_length": 1024}
    connectors_dir = tmp_path / "refiner" / "connectors"
    connectors_dir.mkdir(parents=True)
    connector_config = {
        "caption_channels": 8,
        "text_proj_in_factor": 3,
        "video_connector_num_layers": 2,
    }
    (connectors_dir / "config.json").write_text(
        json.dumps(connector_config),
        encoding="utf-8",
    )

    connector_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.refiner_text_connector_builder"
    )

    class FakeShape:
        num_layers = 2

    def fake_shape(config, conn_config):  # noqa: ANN001
        captured["shape_config"] = config
        captured["shape_connector_config"] = conn_config
        return FakeShape()

    def fake_load(path, *, num_layers, precision):
        captured["load_path"] = path
        captured["load_layers"] = num_layers
        captured["load_precision"] = precision
        return {"text_proj_in.weight": "trt-layout-weight"}

    def fake_build(weights, config, conn_config, *, precision, verbose):
        captured["build_weights"] = weights
        captured["build_config"] = config
        captured["build_connector_config"] = conn_config
        captured["build_precision"] = precision
        captured["build_verbose"] = verbose
        return b"refiner-text-connector-plan"

    monkeypatch.setattr(
        connector_builder_mod,
        "refiner_text_connector_shape_from_config",
        fake_shape,
    )
    monkeypatch.setattr(
        connector_builder_mod,
        "load_sana_wm_refiner_text_connector_weights",
        fake_load,
    )
    monkeypatch.setattr(
        connector_builder_mod,
        "build_sana_wm_refiner_text_connector_engine",
        fake_build,
    )

    plan = sana_wm_plugin_mod._build_sana_wm_refiner_text_connector_plan(
        connectors_dir,
        raw_config,
        connector_config,
        precision="fp16",
        verbose=True,
    )

    assert plan == b"refiner-text-connector-plan"
    assert captured["shape_config"] is raw_config
    assert captured["shape_connector_config"] is connector_config
    assert captured["load_path"] == connectors_dir
    assert captured["load_layers"] == 2
    assert captured["load_precision"] == "fp16"
    assert captured["build_weights"] == {"text_proj_in.weight": "trt-layout-weight"}
    assert captured["build_config"] is raw_config
    assert captured["build_connector_config"] is connector_config
    assert captured["build_precision"] == "fp16"
    assert captured["build_verbose"] is True


def test_sana_wm_refiner_text_connector_accepts_public_split_rope() -> None:
    connector_builder_mod = importlib.import_module(
        "tensorrt_model_connect.families.sana_wm.refiner_text_connector_builder"
    )
    shape = connector_builder_mod.refiner_text_connector_shape_from_config(
        {"sana_wm_refiner_text_max_length": 4},
        {
            "rope_type": "split",
            "video_connector_num_attention_heads": 2,
            "video_connector_attention_head_dim": 4,
        },
    )

    cos, sin = connector_builder_mod._make_text_rope_tables(shape)

    assert shape.rope_type == "split"
    assert cos.shape == (4, 8)
    assert sin.shape == (4, 8)
    np.testing.assert_allclose(cos[:, 0:2], cos[:, 2:4])
    np.testing.assert_allclose(cos[:, 4:6], cos[:, 6:8])
    np.testing.assert_allclose(sin[:, 0:2], sin[:, 2:4])
    np.testing.assert_allclose(sin[:, 4:6], sin[:, 6:8])


def test_sana_wm_plugin_builds_stage1_denoiser_extra_plan_from_raw_dit(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    dit_path = tmp_path / "dit" / "sana_wm_1600m_720p.safetensors"
    _write_safetensors_header(
        dit_path,
        {
            "x_embedder.proj.weight": ("F32", [2240, 128, 1, 1, 1]),
            "y_embedder.y_proj.fc1.weight": ("F32", [2240, 2304]),
            "y_embedder.y_embedding": ("BF16", [300, 2304]),
            "plucker_embedder.proj.weight": ("F32", [2240, 48, 1, 1, 1]),
            "raymap_embedder.proj.weight": ("F32", [2240, 3, 1, 1, 1]),
            "blocks.0.attn.qkv.weight": ("F32", [6720, 2240]),
            "final_layer.linear.weight": ("F32", [128, 2240]),
        },
    )
    captured: dict[str, object] = {}

    def fake_build(dit, raw_config, *, precision, verbose):  # noqa: ANN001
        captured["dit"] = dit
        captured["raw_config"] = raw_config
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"stage1-denoiser-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_stage1_denoiser_plan",
        fake_build,
    )
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    plans = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        256,
        precision="fp16",
        verbose=True,
    )

    assert plans["denoiser_plan"] == b"stage1-denoiser-plan"
    assert captured["dit"] == dit_path
    assert captured["raw_config"] is cfg.raw
    assert captured["precision"] == "fp16"
    assert captured["verbose"] is True


def test_sana_wm_plugin_builds_missing_refiner_denoiser_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path, include_refiner_denoiser=False)
    _write_tokenizer(tmp_path)
    _write_refiner_diffusers_markers(tmp_path)
    transformer_dir = tmp_path / "refiner" / "transformer"
    (transformer_dir / "config.json").write_text(
        json.dumps({"num_layers": 2, "num_attention_heads": 4, "attention_head_dim": 8}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build(transformer, raw_config, transformer_config, *, precision, verbose):
        captured["transformer"] = transformer
        captured["raw_config"] = raw_config
        captured["transformer_config"] = transformer_config
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-refiner-denoiser-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_refiner_denoiser_plan",
        fake_build,
    )
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_refiner_denoiser_plan" not in weights["_native_plan_paths"]
    assert weights["_refiner_transformer_dir"].endswith("refiner/transformer")
    assert weights["_can_build_refiner_denoiser_plan"] is True
    assert sana_wm_mod.build_engine(cfg, weights, 256) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        256,
        precision="fp16",
        verbose=True,
    )

    assert extras["sana_wm_refiner_denoiser_plan"] == b"generated-refiner-denoiser-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["transformer"] == transformer_dir
    assert captured["raw_config"] is cfg.raw
    assert captured["transformer_config"] == {
        "num_layers": 2,
        "num_attention_heads": 4,
        "attention_head_dim": 8,
    }
    assert captured["precision"] == "fp16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "sana_wm_refiner_denoiser_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_reads_local_stage1_dit_metadata(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_safetensors_header(
        tmp_path / "dit" / "sana_wm_1600m_720p.safetensors",
        {
            "x_embedder.proj.weight": ("F32", [2240, 128, 1, 1, 1]),
            "y_embedder.y_proj.fc1.weight": ("F32", [2240, 2304]),
            "y_embedder.y_embedding": ("BF16", [300, 2304]),
            "final_layer.linear.weight": ("F32", [128, 2240]),
            "plucker_embedder.proj.weight": ("F32", [2240, 48, 1, 1, 1]),
            "raymap_embedder.proj.weight": ("F32", [2240, 3, 1, 1, 1]),
            "blocks.0.attn.qkv.weight": ("F32", [6720, 2240]),
            "blocks.1.attn.qkv.weight": ("F32", [6720, 2240]),
        },
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_stage1_dit_path"].endswith("dit/sana_wm_1600m_720p.safetensors")
    assert weights["_vae_dir"].endswith("vae")
    assert weights["_refiner_checkpoint"].endswith("refiner")
    summary = weights["_stage1_dit_summary"]
    assert summary["num_layers"] == 2
    assert summary["hidden_size"] == 2240
    assert summary["latent_channels"] == 128
    assert summary["text_max_length"] == 300
    assert summary["text_embed_dim"] == 2304
    assert summary["chunk_plucker_channels"] == 48
    assert summary["raymap_channels"] == 3

    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert overrides["sana_wm_dit_num_layers"] == 2
    assert overrides["sana_wm_dit_hidden_size"] == 2240
    assert overrides["sana_wm_dit_text_embed_dim"] == 2304
    assert overrides["sana_wm_dit_tensor_count"] == 8


def test_sana_wm_plugin_accepts_legacy_single_file_refiner_layout(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    refiner_dir = tmp_path / "refiner"
    refiner_dir.mkdir()
    (refiner_dir / "refiner.safetensors").write_bytes(b"legacy-refiner")

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_refiner_checkpoint"].endswith("refiner/refiner.safetensors")
    assert weights["_refiner_transformer_dir"].endswith("refiner")
    assert weights["_refiner_connectors_dir"].endswith("refiner")


def test_sana_wm_plugin_embeds_prebuilt_native_plan_sections(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path)
    _write_tokenizer(tmp_path)
    (tmp_path / "text_encoder" / "tokenizer_config.json").write_text(
        '{"add_bos_token": true}',
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"].endswith("denoiser_plan.plan")
    assert weights["_native_plan_paths"]["sana_wm_vae_encoder_plan"].endswith(
        "sana_wm_vae_encoder_plan.plan"
    )
    assert weights["_tokenizer_sections"]["tokenizer.json"].endswith("text_encoder/tokenizer.json")
    assert weights["_tokenizer_sections"]["sana_wm_stage1_tokenizer.json"].endswith(
        "text_encoder/tokenizer.json"
    )
    assert weights["_tokenizer_sections"]["sana_wm_refiner_tokenizer.json"].endswith(
        "refiner/text_encoder/tokenizer.json"
    )

    extras = sana_wm_mod.build_extra_engines(cfg, weights, 256)
    for section, data in plans.items():
        assert extras[section] == data
    assert extras["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    assert extras["sana_wm_stage1_tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    assert (
        extras["sana_wm_refiner_tokenizer.json"]
        == b'{"model": {"type": "Unigram"}, "refiner": true}'
    )
    assert extras["tokenizer_config.json"] == b'{"add_bos_token": true}'
    assert sana_wm_mod.build_engine(cfg, weights, 256) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"

    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert overrides["sana_wm_native_plan_sections"] == list(plans)


def test_sana_wm_plugin_accepts_segmented_stage1_denoiser_plan_set(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(
        tmp_path,
        include_stage1_denoiser=False,
        include_stage1_segments=True,
    )
    _write_tokenizer(tmp_path)

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "denoiser_plan" not in weights["_native_plan_paths"]
    for section in sana_wm_plugin_mod._STAGE1_SEGMENTED_DENOISER_SECTIONS:
        assert weights["_native_plan_paths"][section].endswith(f"{section}.plan")

    extras = sana_wm_mod.build_extra_engines(cfg, weights, 256)
    for section, data in plans.items():
        assert extras[section] == data

    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    sections = overrides["sana_wm_native_plan_sections"]
    assert "denoiser_plan" not in sections
    for section in sana_wm_plugin_mod._STAGE1_SEGMENTED_DENOISER_SECTIONS:
        assert section in sections


def test_sana_wm_plugin_embeds_prebuilt_refiner_text_connector_plan(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path, include_refiner_text_connector=True)
    _write_tokenizer(tmp_path)

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_native_plan_paths"]["sana_wm_refiner_text_connector_plan"].endswith(
        "sana_wm_refiner_text_connector_plan.plan"
    )
    extras = sana_wm_mod.build_extra_engines(cfg, weights, 256)
    assert extras["sana_wm_refiner_text_connector_plan"] == b"refiner-text-connector-plan"
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "sana_wm_refiner_text_connector_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_prefers_stage1_tokenizer_over_refiner_tokenizer(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path)

    stage1_text_encoder = tmp_path / "external-stage1-text-encoder"
    stage1_text_encoder.mkdir()
    (stage1_text_encoder / "config.json").write_text(
        '{"model_type": "gemma"}',
        encoding="utf-8",
    )
    (stage1_text_encoder / "tokenizer.json").write_text(
        '{"stage1": true}',
        encoding="utf-8",
    )
    (stage1_text_encoder / "tokenizer_config.json").write_text(
        '{"add_bos_token": true}',
        encoding="utf-8",
    )
    refiner_text_encoder = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder.mkdir(parents=True)
    (refiner_text_encoder / "config.json").write_text(
        '{"model_type": "gemma3"}',
        encoding="utf-8",
    )
    (refiner_text_encoder / "tokenizer.json").write_text(
        '{"refiner": true}',
        encoding="utf-8",
    )

    monkeypatch.setenv("SANA_WM_TEXT_ENCODER_DIR", str(stage1_text_encoder))

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)
    extras = sana_wm_mod.build_extra_engines(cfg, weights, 256)

    assert extras["tokenizer.json"] == b'{"stage1": true}'
    assert extras["tokenizer_config.json"] == b'{"add_bos_token": true}'
    assert extras["sana_wm_stage1_tokenizer.json"] == b'{"stage1": true}'
    assert extras["sana_wm_refiner_tokenizer.json"] == b'{"refiner": true}'
    assert cfg.raw["tokenizer_add_special_tokens"] == 1


def test_sana_wm_tokenizer_policy_uses_encode_behavior(monkeypatch, tmp_path) -> None:
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens=True):
            del text
            return [2, 7] if add_special_tokens else [7]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert path == tmp_path
            assert local_files_only is True
            return FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)

    assert sana_wm_plugin_mod._tokenizer_adds_special_tokens(tmp_path) is True


def test_sana_wm_plugin_builds_missing_stage1_text_encoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path, include_text_encoder=False)
    _write_tokenizer(tmp_path)
    _write_text_encoder_config(tmp_path)
    captured: dict[str, object] = {}

    def fake_build_text_encoder_plan(text_encoder_dir, max_cache_length, *, precision, verbose):
        captured["text_encoder_dir"] = text_encoder_dir
        captured["max_cache_length"] = max_cache_length
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-stage1-text-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_stage1_text_encoder_plan",
        fake_build_text_encoder_plan,
    )
    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_stage1_chi_prompt_token_count",
        lambda text_encoder_dir, chi_prompt: 12,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "text_encoder_0_plan" not in weights["_native_plan_paths"]
    assert weights["_stage1_text_encoder_dir"].endswith("text_encoder")
    assert (
        sana_wm_mod.build_engine(
            cfg,
            weights,
            512,
            precision="bf16",
        )
        == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    )

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        512,
        precision="bf16",
        verbose=True,
    )

    assert extras["text_encoder_0_plan"] == b"generated-stage1-text-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["text_encoder_dir"] == tmp_path / "text_encoder"
    assert captured["max_cache_length"] == 310
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert overrides["sana_wm_native_plan_sections"][0] == "text_encoder_0_plan"
    assert set(overrides["sana_wm_native_plan_sections"]) == set(plans) | {"text_encoder_0_plan"}


def test_sana_wm_plugin_builds_missing_refiner_text_encoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path, include_refiner_text_encoder=False)
    _write_tokenizer(tmp_path)
    refiner_text_encoder_dir = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (refiner_text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 16,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build_text_encoder_plan(text_encoder_dir, max_cache_length, *, precision, verbose):
        captured["text_encoder_dir"] = text_encoder_dir
        captured["max_cache_length"] = max_cache_length
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-refiner-text-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_refiner_text_encoder_plan",
        fake_build_text_encoder_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_refiner_text_encoder_plan" not in weights["_native_plan_paths"]
    assert weights["_refiner_text_encoder_dir"].endswith("refiner/text_encoder")
    assert (
        sana_wm_mod.build_engine(
            cfg,
            weights,
            128,
            precision="bf16",
        )
        == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    )

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        128,
        precision="bf16",
        verbose=True,
    )

    assert extras["sana_wm_refiner_text_encoder_plan"] == b"generated-refiner-text-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["text_encoder_dir"] == refiner_text_encoder_dir
    assert captured["max_cache_length"] == 1024
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "sana_wm_refiner_text_encoder_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_builds_split_gemma3_refiner_text_encoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(
        tmp_path,
        include_refiner_text_encoder=False,
        include_refiner_text_connector=True,
    )
    _write_tokenizer(tmp_path)
    refiner_text_encoder_dir = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (refiner_text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "text_config": {"model_type": "gemma3_text"},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build_text_encoder_plan(text_encoder_dir, max_cache_length, *, precision, verbose):
        captured["text_encoder_dir"] = text_encoder_dir
        captured["max_cache_length"] = max_cache_length
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-split-refiner-text-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_refiner_text_encoder_plan",
        fake_build_text_encoder_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_refiner_text_encoder_plan" not in weights["_native_plan_paths"]
    assert weights["_can_build_refiner_text_encoder_plan"] is True
    assert sana_wm_mod.build_engine(cfg, weights, 128) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        128,
        precision="bf16",
        verbose=True,
    )

    assert extras["sana_wm_refiner_text_encoder_plan"] == b"generated-split-refiner-text-plan"
    assert (
        extras["sana_wm_refiner_text_connector_plan"]
        == plans["sana_wm_refiner_text_connector_plan"]
    )
    assert captured["text_encoder_dir"] == refiner_text_encoder_dir
    assert captured["max_cache_length"] == 1024
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert set(overrides["sana_wm_native_plan_sections"]) == set(plans) | {
        "sana_wm_refiner_text_encoder_plan"
    }


def test_sana_wm_plugin_builds_split_refiner_text_connector_plan_from_raw(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(
        tmp_path,
        include_refiner_text_encoder=False,
        include_refiner_text_connector=False,
    )
    _write_tokenizer(tmp_path)
    connectors_dir = tmp_path / "refiner" / "connectors"
    connectors_dir.mkdir(parents=True, exist_ok=True)
    (connectors_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"refiner-connectors")
    connector_config = {
        "caption_channels": 8,
        "text_proj_in_factor": 3,
        "video_connector_num_layers": 2,
    }
    (connectors_dir / "config.json").write_text(
        json.dumps(connector_config),
        encoding="utf-8",
    )
    refiner_text_encoder_dir = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (refiner_text_encoder_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "text_config": {"model_type": "gemma3_text"},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_build_text_encoder_plan(text_encoder_dir, max_cache_length, *, precision, verbose):
        captured["text_encoder_dir"] = text_encoder_dir
        captured["text_max_cache_length"] = max_cache_length
        captured["text_precision"] = precision
        captured["text_verbose"] = verbose
        return b"generated-split-refiner-text-plan"

    def fake_build_connector_plan(
        conn_dir,
        raw_config,
        conn_config,
        *,
        precision,
        verbose,
    ):
        captured["connectors_dir"] = conn_dir
        captured["connector_raw_config"] = raw_config
        captured["connector_config"] = conn_config
        captured["connector_precision"] = precision
        captured["connector_verbose"] = verbose
        return b"generated-refiner-text-connector-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_refiner_text_encoder_plan",
        fake_build_text_encoder_plan,
    )
    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_refiner_text_connector_plan",
        fake_build_connector_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_refiner_text_encoder_plan" not in weights["_native_plan_paths"]
    assert "sana_wm_refiner_text_connector_plan" not in weights["_native_plan_paths"]
    assert weights["_can_build_refiner_text_encoder_plan"] is True
    assert weights["_can_build_refiner_text_connector_plan"] is True

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        128,
        precision="bf16",
        verbose=True,
    )

    assert extras["sana_wm_refiner_text_encoder_plan"] == b"generated-split-refiner-text-plan"
    assert extras["sana_wm_refiner_text_connector_plan"] == b"generated-refiner-text-connector-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["text_encoder_dir"] == refiner_text_encoder_dir
    assert captured["connectors_dir"] == connectors_dir
    assert captured["connector_raw_config"] is cfg.raw
    assert captured["connector_config"] == connector_config
    assert captured["connector_precision"] == "bf16"
    assert captured["connector_verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "sana_wm_refiner_text_encoder_plan" in overrides["sana_wm_native_plan_sections"]
    assert "sana_wm_refiner_text_connector_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_builds_split_gemma_refiner_text_with_connectors(
    tmp_path,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path, include_refiner_text_encoder=False)
    _write_tokenizer(tmp_path)
    _write_refiner_diffusers_markers(tmp_path)
    (tmp_path / "refiner" / "text_encoder" / "config.json").write_text(
        json.dumps({"model_type": "gemma"}),
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert weights["_refiner_text_encoder_model_type"] == "gemma"
    assert weights["_can_build_refiner_text_encoder_plan"] is True
    assert weights["_can_build_refiner_text_connector_plan"] is True


def test_sana_wm_plugin_builds_missing_vae_decoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path)
    (tmp_path / "trtmc_engines" / "vae_decoder_plan.plan").unlink()
    plans.pop("vae_decoder_plan")
    _write_tokenizer(tmp_path)
    _write_vae_weights_marker(tmp_path)
    captured: dict[str, object] = {}

    def fake_build_vae_decoder_plan(vae_dir, raw_config, *, precision, verbose):
        captured["vae_dir"] = vae_dir
        captured["raw_config"] = raw_config
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-vae-decoder-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_vae_decoder_plan",
        fake_build_vae_decoder_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "vae_decoder_plan" not in weights["_native_plan_paths"]
    assert weights["_sana_wm_vae_decoder_dir"].endswith("vae")
    assert sana_wm_mod.build_engine(cfg, weights, 256) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        256,
        precision="fp16",
        verbose=True,
    )

    assert extras["vae_decoder_plan"] == b"generated-vae-decoder-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["vae_dir"] == tmp_path / "vae"
    assert captured["raw_config"] is cfg.raw
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "vae_decoder_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_vae_decoder_plan_uses_decoder_spatial_padding_mode(
    tmp_path,
    monkeypatch,
) -> None:
    from tensorrt_model_connect.families.sana_wm.components.ltx_video import ltx_vae_builder

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps({"decoder_spatial_padding_mode": "reflect"}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_load_ltx_vae_weights(model_dir, *, precision):
        captured["load_model_dir"] = model_dir
        captured["load_precision"] = precision
        return {}

    def fake_build_ltx_vae_decoder_engine(weights, **kwargs):
        captured["weights"] = weights
        captured.update(kwargs)
        return b"decoder-plan"

    monkeypatch.setattr(
        ltx_vae_builder,
        "load_ltx_vae_weights",
        fake_load_ltx_vae_weights,
    )
    monkeypatch.setattr(
        ltx_vae_builder,
        "build_ltx_vae_decoder_engine",
        fake_build_ltx_vae_decoder_engine,
    )

    raw_config = {
        "video_height": 704,
        "video_width": 1280,
        "video_num_frames": 321,
        "vae": {
            "vae_latent_dim": 128,
            "vae_stride": [8, 32, 32],
        },
    }

    plan = sana_wm_plugin_mod._build_sana_wm_vae_decoder_plan(
        vae_dir,
        raw_config,
        precision="bf16",
        verbose=True,
    )

    assert plan == b"decoder-plan"
    assert captured["load_model_dir"] == vae_dir
    assert captured["load_precision"] == "bf16"
    assert captured["latent_frames"] == 41
    assert captured["latent_height"] == 22
    assert captured["latent_width"] == 40
    assert captured["spatial_padding_mode"] == "reflect"
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True


def test_sana_wm_vae_tile_shapes_match_pr379_model_card() -> None:
    raw_config = {
        "video_height": 704,
        "video_width": 1280,
        "video_num_frames": 321,
        "vae": {
            "vae_latent_dim": 128,
            "vae_stride": [8, 32, 32],
            "use_framewise_decoding": True,
            "tile_sample_min_height": 512,
            "tile_sample_min_width": 512,
            "tile_sample_stride_height": 448,
            "tile_sample_stride_width": 448,
            "tile_sample_min_num_frames": 96,
            "tile_sample_stride_num_frames": 64,
        },
    }

    shapes = sana_wm_plugin_mod._sana_wm_vae_tile_shapes(raw_config)

    assert shapes == [
        (9, 8, 12),
        (9, 8, 16),
        (9, 16, 12),
        (9, 16, 16),
        (13, 8, 12),
        (13, 8, 16),
        (13, 16, 12),
        (13, 16, 16),
    ]


def test_ltx_vae_plugin_helpers_resolve_family_stage1_builder(monkeypatch) -> None:
    from tensorrt_model_connect.families.sana_wm import stage1_dit_builder
    from tensorrt_model_connect.families.sana_wm.components.ltx_video import ltx_vae_builder

    trt_module = object()
    resolved_plugins: list[tuple[object, str]] = []

    def missing_plugin(module, name):
        resolved_plugins.append((module, name))
        return None

    monkeypatch.setattr(ltx_vae_builder, "_ensure_trt", lambda: trt_module)
    monkeypatch.setattr(
        stage1_dit_builder,
        "_get_sana_wm_plugin_creator_with_get_creator",
        missing_plugin,
    )

    calls = [
        (
            "SanaWmTorchConv3d",
            lambda: ltx_vae_builder._add_torch_conv3d(
                None,
                None,
                np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
                None,
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            ),
        ),
        (
            "SanaWmVaeRmsSilu",
            lambda: ltx_vae_builder._add_vae_rms_silu(None, None, eps=1e-6),
        ),
        (
            "SanaWmVaeDenormalize",
            lambda: ltx_vae_builder._add_vae_denormalize(
                None,
                None,
                {},
                1,
                scaling_factor=1.0,
            ),
        ),
        (
            "SanaWmVaeLayerNorm",
            lambda: ltx_vae_builder._add_vae_layer_norm(
                None,
                None,
                channels=1,
                weight=np.ones(1, dtype=np.float32),
                bias=np.zeros(1, dtype=np.float32),
                eps=1e-6,
            ),
        ),
    ]

    for plugin_name, call in calls:
        with pytest.raises(RuntimeError, match=plugin_name):
            call()

    assert resolved_plugins == [(trt_module, name) for name, _call in calls]


def test_ltx_vae_tiled_encoder_reuses_blended_neighbors(monkeypatch) -> None:
    from tensorrt_model_connect.families.sana_wm.components.ltx_video import ltx_vae_builder

    class FakeTensor:
        def __init__(self, shape, name: str):
            self.shape = tuple(shape)
            self.name = name
            self.dtype = "float32"

    class FakeLayer:
        def __init__(self, output: FakeTensor):
            self._output = output

        def get_output(self, _index: int) -> FakeTensor:
            return self._output

    class FakeConcatLayer:
        def __init__(self, tensors: list[FakeTensor]):
            self._tensors = tensors
            self.axis = 0

        def get_output(self, _index: int) -> FakeTensor:
            shape = list(self._tensors[0].shape)
            shape[self.axis] = sum(int(t.shape[self.axis]) for t in self._tensors)
            names = ",".join(t.name for t in self._tensors)
            return FakeTensor(tuple(shape), f"concat({names})")

    class FakeNetwork:
        def add_slice(self, inp, start, size, _stride):
            if getattr(inp, "name", "") == "sample":
                name = f"tile_{start[3]}_{start[4]}"
            else:
                name = f"slice({inp.name})"
            return FakeLayer(FakeTensor(size, name))

        def add_concatenation(self, tensors):
            return FakeConcatLayer(tensors)

    def fake_encoder_body(_network, tile, _weights, **kwargs):
        return (
            FakeTensor(
                (1, 1, 1, kwargs["sample_height"], kwargs["sample_width"]),
                tile.name,
            ),
            1,
            kwargs["sample_height"],
            kwargs["sample_width"],
        )

    vertical_calls = []
    horizontal_calls = []

    def fake_blend_v(_network, above, current, _blend_extent):
        vertical_calls.append((above, current))
        return FakeTensor(current.shape, f"v({above.name},{current.name})")

    def fake_blend_h(_network, left, current, _blend_extent):
        horizontal_calls.append((left, current))
        return FakeTensor(current.shape, f"h({left.name},{current.name})")

    monkeypatch.setattr(ltx_vae_builder, "_ltx_vae_encoder_body", fake_encoder_body)
    monkeypatch.setattr(ltx_vae_builder, "_ltx_vae_blend_v", fake_blend_v)
    monkeypatch.setattr(ltx_vae_builder, "_ltx_vae_blend_h", fake_blend_h)

    ltx_vae_builder._ltx_vae_encoder_tiled(
        FakeNetwork(),
        FakeTensor((1, 3, 1, 4, 4), "sample"),
        {},
        sample_frames=1,
        sample_height=4,
        sample_width=4,
        in_channels=3,
        latent_channels=1,
        block_out_channels=(1,),
        layers_per_block=(1,),
        spatio_temporal_scaling=(),
        downsample_type=(),
        patch_size=1,
        patch_size_t=1,
        dtype=np.float32,
        tile_sample_min_height=3,
        tile_sample_min_width=3,
        tile_sample_stride_height=2,
        tile_sample_stride_width=2,
        verbose=False,
    )

    assert vertical_calls[1][0].name.startswith("h(")
    assert horizontal_calls[1][0].name.startswith("v(")


def test_sana_wm_plugin_builds_missing_refiner_vae_decoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path, include_refiner_vae_decoder=False)
    _write_tokenizer(tmp_path)
    _write_vae_weights_marker(tmp_path)
    captured: dict[str, object] = {}

    def fake_build_refiner_vae_decoder_plan(vae_dir, raw_config, *, precision, verbose):
        captured["vae_dir"] = vae_dir
        captured["raw_config"] = raw_config
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-refiner-vae-decoder-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_refiner_vae_decoder_plan",
        fake_build_refiner_vae_decoder_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_refiner_vae_decoder_plan" not in weights["_native_plan_paths"]
    assert weights["_sana_wm_refiner_vae_decoder_dir"].endswith("vae")
    assert sana_wm_mod.build_engine(cfg, weights, 256) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        256,
        precision="fp16",
        verbose=True,
    )

    assert extras["sana_wm_refiner_vae_decoder_plan"] == b"generated-refiner-vae-decoder-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["vae_dir"] == tmp_path / "vae"
    assert captured["raw_config"] is cfg.raw
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "sana_wm_refiner_vae_decoder_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_does_not_force_refiner_for_stage1_only_bundle(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(
        tmp_path,
        include_refiner_text_encoder=False,
        include_refiner_denoiser=False,
        include_refiner_vae_decoder=False,
    )
    _write_tokenizer(tmp_path)
    _write_vae_weights_marker(tmp_path)
    refiner_text_encoder_dir = tmp_path / "refiner" / "text_encoder"
    refiner_text_encoder_dir.mkdir(parents=True, exist_ok=True)
    (refiner_text_encoder_dir / "config.json").write_text(
        json.dumps({"model_type": "gemma"}),
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert sana_wm_mod.build_engine(cfg, weights, 256) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")
    extras = sana_wm_mod.build_extra_engines(cfg, weights, 256)

    for section, data in plans.items():
        assert extras[section] == data
    assert "sana_wm_refiner_text_encoder_plan" not in extras
    assert "sana_wm_refiner_vae_decoder_plan" not in extras
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "sana_wm_refiner_text_encoder_plan" not in overrides["sana_wm_native_plan_sections"]
    assert "sana_wm_refiner_vae_decoder_plan" not in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_builds_missing_vae_encoder_plan(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(tmp_path, include_vae_encoder=False)
    _write_tokenizer(tmp_path)
    _write_vae_weights_marker(tmp_path)
    captured: dict[str, object] = {}

    def fake_build_vae_encoder_plan(vae_dir, raw_config, *, precision, verbose):
        captured["vae_dir"] = vae_dir
        captured["raw_config"] = raw_config
        captured["precision"] = precision
        captured["verbose"] = verbose
        return b"generated-vae-encoder-plan"

    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_build_sana_wm_vae_encoder_plan",
        fake_build_vae_encoder_plan,
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.load_weights(str(tmp_path), cfg)

    assert "sana_wm_vae_encoder_plan" not in weights["_native_plan_paths"]
    assert weights["_sana_wm_vae_encoder_dir"].endswith("vae")
    assert sana_wm_mod.build_engine(cfg, weights, 256) == (b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n")

    extras = sana_wm_mod.build_extra_engines(
        cfg,
        weights,
        256,
        precision="fp16",
        verbose=True,
    )

    assert extras["sana_wm_vae_encoder_plan"] == b"generated-vae-encoder-plan"
    for section, data in plans.items():
        assert extras[section] == data
    assert captured["vae_dir"] == tmp_path / "vae"
    assert captured["raw_config"] is cfg.raw
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert "sana_wm_vae_encoder_plan" in overrides["sana_wm_native_plan_sections"]


def test_sana_wm_plugin_discovers_native_plan_dir_from_config(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    plan_dir = tmp_path / "native_plans"
    plans = _write_native_plan_set(model_dir, plan_dir=plan_dir)
    _write_tokenizer(model_dir)
    (model_dir / "config.yaml").write_text(
        _sana_yaml() + f"\nsana_wm_native_plan_dir: {plan_dir}\n",
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(model_dir)
    weights = sana_wm_mod.load_weights(str(model_dir), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"] == str(plan_dir / "denoiser_plan.plan")
    assert sana_wm_mod.build_engine(cfg, weights, 256) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    assert (
        sana_wm_mod.build_extra_engines(cfg, weights, 256)["denoiser_plan"]
        == plans["denoiser_plan"]
    )


def test_sana_wm_plugin_discovers_native_plan_dir_from_env(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    plan_dir = tmp_path / "env_native_plans"
    _write_native_plan_set(model_dir, plan_dir=plan_dir)
    _write_tokenizer(model_dir)
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    monkeypatch.setenv("SANA_WM_NATIVE_PLAN_DIR", str(plan_dir))

    cfg = ModelConfig.from_dir(model_dir)
    weights = sana_wm_mod.load_weights(str(model_dir), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"] == str(plan_dir / "denoiser_plan.plan")
    assert sana_wm_mod.build_engine(cfg, weights, 256) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"


def test_sana_wm_plugin_discovers_native_plan_model_dir_from_config(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    external_model_dir = tmp_path / "native_model"
    plans = _write_native_plan_set(external_model_dir)
    _write_tokenizer(model_dir)
    (model_dir / "config.yaml").write_text(
        _sana_yaml() + f"\nsana_wm_model_dir: {external_model_dir}\n",
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(model_dir)
    weights = sana_wm_mod.load_weights(str(model_dir), cfg)

    native_plan_paths = weights["_native_plan_paths"]
    assert native_plan_paths["denoiser_plan"] == str(
        external_model_dir / "trtmc_engines" / "denoiser_plan.plan"
    )
    assert (
        sana_wm_mod.build_extra_engines(cfg, weights, 256)["denoiser_plan"]
        == plans["denoiser_plan"]
    )


def test_sana_wm_plugin_rejects_partial_native_plan_sections(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    engine_dir = tmp_path / "trtmc_engines"
    engine_dir.mkdir()
    (engine_dir / "denoiser_plan.plan").write_bytes(b"stage1-dit-plan")

    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="complete prebuilt plan set"):
        sana_wm_mod.load_weights(str(tmp_path), cfg)


def test_sana_wm_plugin_rejects_native_plan_sections_without_tokenizer(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path)

    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="require tokenizer assets"):
        sana_wm_mod.load_weights(str(tmp_path), cfg)


def test_sana_wm_build_bundle_rejects_missing_native_sections(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    output_path = str(tmp_path / "sana-wm.bundle")

    with pytest.raises(
        NotImplementedError,
        match="pure C\\+\\+ builds require a complete native",
    ):
        sana_wm_mod.build(
            str(model_dir),
            output_path,
            build_timing_path=str(tmp_path / "timing.json"),
        )


def test_sana_wm_build_bundle_embeds_native_sections(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    plans = _write_native_plan_set(model_dir)
    _write_tokenizer(model_dir)
    (model_dir / "text_encoder" / "tokenizer_config.json").write_text(
        '{"add_bos_token": true}',
        encoding="utf-8",
    )
    output_path = str(tmp_path / "sana-wm.bundle")
    captured: dict[str, object] = {}

    def fake_write_bundle(path, info, sections):
        captured["path"] = path
        captured["info"] = info
        captured["sections"] = sections

    from tensorrt_model_connect import bundle_writer
    from tensorrt_model_connect.tvm_ffi import graph_build
    from tensorrt_model_connect.families.sana_wm import native_plugin_builder

    native_plugin = tmp_path / "sana_wm_native_plugin.so"
    native_plugin.write_bytes(b"native-plugin")

    monkeypatch.setattr(bundle_writer, "tensorrt_version", lambda: "10.0.0")
    monkeypatch.setattr(bundle_writer, "tensorrt_abi", lambda _version: "")
    monkeypatch.setattr(bundle_writer, "gpu_name", lambda: "mock-gpu")
    monkeypatch.setattr(bundle_writer, "write_bundle", fake_write_bundle)
    monkeypatch.setattr(graph_build, "kernel_slots_section", lambda: None)
    monkeypatch.setattr(
        native_plugin_builder,
        "ensure_native_plugin",
        lambda **_kwargs: native_plugin,
    )
    monkeypatch.setattr(sana_wm_mod.ctypes, "CDLL", lambda *_args, **_kwargs: None)

    sana_wm_mod.build(
        str(model_dir),
        output_path,
        build_timing_path=str(tmp_path / "timing.json"),
    )

    sections = {section.name: section.data for section in captured["sections"]}
    assert sections["engine_plan"] == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    for section, data in plans.items():
        assert sections[section] == data
    assert sections["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    config = json.loads(sections["config.json"].decode("utf-8"))
    assert config["engine_backend"] == "trt"
    assert config["tokenizer_add_special_tokens"] == 1
    assert "sana_wm_allow_python_bridge" not in config
    assert config["sana_wm_native_plan_sections"] == list(plans)


def test_sana_wm_pre_scales_gemma_embedding_in_bf16_without_changing_lm_head() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    embedding = np.asarray(
        [[0.10009765625, -0.0311279296875], [0.00787353515625, 0.333984375]],
        dtype=np.float16,
    )
    lm_head = embedding.T.copy()
    weights = {
        "embedding": embedding.copy(),
        "w_out": lm_head.copy(),
        "_embedding_scale": 48.0,
    }

    sana_wm_plugin_mod._pre_scale_gemma_embedding_bf16(weights, 2304)

    expected = (
        embedding.astype(ml_dtypes.bfloat16) * np.asarray(48.0, dtype=ml_dtypes.bfloat16)
    ).astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(weights["embedding"], expected.astype(np.float16))
    np.testing.assert_array_equal(weights["w_out"], lm_head)
    assert "_embedding_scale" not in weights


def test_sana_wm_exact_gemma_weight_metadata_is_model_local(monkeypatch) -> None:
    cfg = ModelConfig(
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        max_position_embeddings=16,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        raw={"head_dim": 4},
    )
    weights = {
        "embedding": np.ones((8, 4), dtype=np.float16),
        "w_out": np.ones((4, 8), dtype=np.float16),
        "_embedding_scale": 2.0,
    }
    cos = np.arange(32, dtype=np.float16).reshape(8, 4)
    sin = -cos
    monkeypatch.setattr(
        sana_wm_plugin_mod,
        "_make_exact_gemma_rope_tables",
        lambda config, length: (cos, sin),
    )

    sana_wm_plugin_mod._prepare_exact_gemma_text_weights(cfg, weights, 4)

    assert weights["_sana_wm_exact_gemma"] is True
    assert weights["_sana_wm_rope_cos"] is cos
    assert weights["_sana_wm_rope_sin"] is sin
    assert "_embedding_scale" not in weights
