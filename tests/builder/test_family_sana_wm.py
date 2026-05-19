"""Unit tests for SANA-WM family config and plugin behavior."""

from __future__ import annotations

import json
import struct

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect import engine_builder
import tensorrt_model_connect.families.sana_wm as sana_wm_mod


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


def test_sana_wm_yaml_config_parses_from_dir(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")

    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "sana_wm"
    assert cfg.hidden_size == 1792
    assert cfg.num_hidden_layers == 20
    assert cfg.num_attention_heads == 16
    assert cfg.max_position_embeddings == 300
    assert cfg.raw["runtime_strategy"] == "diffusion_sana_wm"
    assert cfg.raw["video_num_frames"] == 321
    assert cfg.raw["video_height"] == 704
    assert cfg.raw["video_width"] == 1280
    assert cfg.raw["sana_wm_config"]["vae"]["vae_latent_dim"] == 128


def test_sana_wm_plugin_emits_bridge_runtime_config(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    cfg = ModelConfig.from_dir(tmp_path)

    plugin = sana_wm_mod.plugin
    assert plugin.matches("sana_wm")
    assert plugin.matches("SanaMSVideoCamCtrl_1600M_P1_D20")
    assert not plugin.matches("ltx_video")

    weights = plugin.load_weights(str(tmp_path), cfg)
    assert weights["_model_format"] == "sana_wm_yaml"
    assert plugin.build_engine(cfg, weights, 256) == b"TRTMC_SANA_WM_PYTHON_BRIDGE\n"

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
    assert overrides["sana_wm_require_official_script"] == 1
    assert overrides["video_num_frames"] == 321
    assert overrides["fps"] == 16
    assert overrides["num_inference_steps"] == 60
    assert overrides["guidance_scale"] == 5.0
    assert overrides["vae_time_stride"] == 8
    assert overrides["vae_spatial_stride"] == 32
    assert overrides["flow_shift"] == 9.8
    assert overrides["text_encoder_name"] == "gemma-2-2b-it"
    assert overrides["text_encoder_max_length"] == 300
    assert overrides["sana_wm_chi_prompt"] == 'Generate an "Enhanced prompt".\nUser Prompt: '


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
    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    assert weights["_stage1_dit_path"].endswith("dit/sana_wm_1600m_720p.safetensors")
    assert weights["_vae_dir"].endswith("vae")
    assert weights["_refiner_checkpoint"].endswith("refiner/refiner.safetensors")
    summary = weights["_stage1_dit_summary"]
    assert summary["num_layers"] == 2
    assert summary["hidden_size"] == 2240
    assert summary["latent_channels"] == 128
    assert summary["text_max_length"] == 300
    assert summary["text_embed_dim"] == 2304
    assert summary["chunk_plucker_channels"] == 48
    assert summary["raymap_channels"] == 3

    overrides = sana_wm_mod.plugin.get_bundle_config_overrides(cfg)
    assert overrides["sana_wm_dit_num_layers"] == 2
    assert overrides["sana_wm_dit_hidden_size"] == 2240
    assert overrides["sana_wm_dit_text_embed_dim"] == 2304
    assert overrides["sana_wm_dit_tensor_count"] == 8


def test_sana_wm_plugin_embeds_prebuilt_native_plan_sections(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    engine_dir = tmp_path / "trtmc_engines"
    engine_dir.mkdir()
    (engine_dir / "denoiser_plan.plan").write_bytes(b"stage1-dit-plan")
    (engine_dir / "sana_wm_vae_encoder_plan.plan").write_bytes(b"vae-encoder-plan")
    tokenizer_dir = tmp_path / "text_encoder"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"model": {"type": "Unigram"}}', encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text('{"add_bos_token": true}', encoding="utf-8")

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"].endswith("denoiser_plan.plan")
    assert weights["_native_plan_paths"]["sana_wm_vae_encoder_plan"].endswith(
        "sana_wm_vae_encoder_plan.plan"
    )
    assert weights["_tokenizer_sections"]["tokenizer.json"].endswith("text_encoder/tokenizer.json")

    extras = sana_wm_mod.plugin.build_extra_engines(cfg, weights, 256)
    assert extras["denoiser_plan"] == b"stage1-dit-plan"
    assert extras["sana_wm_vae_encoder_plan"] == b"vae-encoder-plan"
    assert extras["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    assert extras["tokenizer_config.json"] == b'{"add_bos_token": true}'

    overrides = sana_wm_mod.plugin.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert overrides["sana_wm_native_plan_sections"] == [
        "denoiser_plan",
        "sana_wm_vae_encoder_plan",
    ]


def test_sana_wm_build_bundle_embeds_bridge_config(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    output_path = str(tmp_path / "sana-wm.trtfb")
    captured: dict[str, object] = {}

    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda rtx=False: None)
    monkeypatch.setattr(
        engine_builder.trt_compat, "resolved_summary", lambda: "mock TensorRT"
    )
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.0.0")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "mock-gpu")

    def fake_write_bundle(path, info, sections):
        captured["path"] = path
        captured["info"] = info
        captured["sections"] = sections

    monkeypatch.setattr(engine_builder, "write_bundle", fake_write_bundle)

    engine_builder.build_bundle(
        str(model_dir),
        output_path,
        build_timing_path=str(tmp_path / "timing.json"),
    )

    assert captured["path"] == output_path
    info = captured["info"]
    assert info.family == "sana_wm"
    assert info.runtime_strategy == "diffusion_sana_wm"

    sections = {section.name: section.data for section in captured["sections"]}
    assert sections["engine_plan"] == b"TRTMC_SANA_WM_PYTHON_BRIDGE\n"
    config = json.loads(sections["config.json"].decode("utf-8"))
    assert config["model_type"] == "sana_wm"
    assert config["runtime_strategy"] == "diffusion_sana_wm"
    assert config["engine_backend"] == "none"
    assert config["sana_wm_hf_id"] == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert config["sana_wm_action"] == "w-80,jw-40,w-40,lw-60,w-100"
    assert config["sana_wm_translation_speed"] == 0.055
    assert config["sana_wm_rotation_speed_deg"] == 1.2
    assert config["sana_wm_default_intrinsics"] == pytest.approx(
        [797.87866, 830.0503, 844.2675, 463.7225]
    )
    assert config["sana_wm_require_official_script"] == 1
    assert config["sana_wm_chi_prompt"] == 'Generate an "Enhanced prompt".\nUser Prompt: '
    assert config["video_num_frames"] == 321
    assert config["vae_time_stride"] == 8
    assert config["vae_spatial_stride"] == 32
    assert config["num_inference_steps"] == 60


def test_sana_wm_build_bundle_embeds_native_sections(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    engine_dir = model_dir / "trtmc_engines"
    engine_dir.mkdir()
    (engine_dir / "denoiser_plan.plan").write_bytes(b"stage1-dit-plan")
    tokenizer_dir = model_dir / "text_encoder"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"model": {"type": "Unigram"}}', encoding="utf-8")
    output_path = str(tmp_path / "sana-wm.trtfb")
    captured: dict[str, object] = {}

    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda rtx=False: None)
    monkeypatch.setattr(
        engine_builder.trt_compat, "resolved_summary", lambda: "mock TensorRT"
    )
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.0.0")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "mock-gpu")

    def fake_write_bundle(path, info, sections):
        captured["path"] = path
        captured["info"] = info
        captured["sections"] = sections

    monkeypatch.setattr(engine_builder, "write_bundle", fake_write_bundle)

    engine_builder.build_bundle(
        str(model_dir),
        output_path,
        build_timing_path=str(tmp_path / "timing.json"),
    )

    sections = {section.name: section.data for section in captured["sections"]}
    assert sections["engine_plan"] == b"TRTMC_SANA_WM_PYTHON_BRIDGE\n"
    assert sections["denoiser_plan"] == b"stage1-dit-plan"
    assert sections["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    config = json.loads(sections["config.json"].decode("utf-8"))
    assert config["engine_backend"] == "trt"
    assert config["sana_wm_native_plan_sections"] == ["denoiser_plan"]
