"""Unit tests for SANA-WM family config and plugin behavior."""

from __future__ import annotations

import importlib
import json
import struct

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect import engine_builder
import tensorrt_model_connect.families.sana_wm as sana_wm_mod

sana_wm_plugin_mod = importlib.import_module(
    "tensorrt_model_connect.families.sana_wm.plugin"
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
) -> dict[str, bytes]:
    engine_dir = plan_dir or model_dir / "trtmc_engines"
    engine_dir.mkdir(parents=True)
    plans = {
        "text_encoder_0_plan": b"text-encoder-plan",
        "denoiser_plan": b"stage1-dit-plan",
        "sana_wm_vae_encoder_plan": b"vae-encoder-plan",
        "vae_decoder_plan": b"stage1-vae-decoder-plan",
        "sana_wm_refiner_text_encoder_plan": b"refiner-text-encoder-plan",
        "sana_wm_refiner_denoiser_plan": b"refiner-denoiser-plan",
        "sana_wm_refiner_vae_decoder_plan": b"refiner-vae-decoder-plan",
    }
    if not include_text_encoder:
        plans.pop("text_encoder_0_plan")
    for section, data in plans.items():
        (engine_dir / f"{section}.plan").write_bytes(data)
    return plans


def _write_tokenizer(model_dir) -> None:
    tokenizer_dir = model_dir / "text_encoder"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text('{"model": {"type": "Unigram"}}', encoding="utf-8")


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


def test_sana_wm_plugin_emits_native_runtime_config(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    cfg = ModelConfig.from_dir(tmp_path)

    plugin = sana_wm_mod.plugin
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
    assert overrides["sana_wm_chi_prompt"] == 'Generate an "Enhanced prompt".\nUser Prompt: '


def test_sana_wm_plugin_rejects_build_without_native_plans(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    cfg = ModelConfig.from_dir(tmp_path)

    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.plugin.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "pure C++ builds require native TensorRT component plans" in message
    assert "SanaMSVideoCamCtrl DiT" in message
    assert "LTX-2 VAE encoder" in message
    assert "TRTMC_SANA_WM_DOWNLOAD_WEIGHTS=1" in message

    overrides = sana_wm_mod.plugin.get_bundle_config_overrides(cfg)
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
    (tmp_path / "refiner").mkdir()
    (tmp_path / "refiner" / "refiner.safetensors").write_bytes(b"placeholder")
    (tmp_path / "refiner" / "text_encoder").mkdir()
    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError) as exc_info:
        sana_wm_mod.plugin.build_engine(cfg, weights, 256)
    message = str(exc_info.value)
    assert "Building those plans directly from raw SANA-WM weights is not implemented yet" in message
    assert "TRTMC_SANA_WM_DOWNLOAD_WEIGHTS" not in message
    assert "stage-1 Gemma text encoder" in message
    assert "complete LTX-2 refiner stack" in message


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
    plans = _write_native_plan_set(tmp_path)
    _write_tokenizer(tmp_path)
    (tmp_path / "text_encoder" / "tokenizer_config.json").write_text(
        '{"add_bos_token": true}',
        encoding="utf-8",
    )

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"].endswith("denoiser_plan.plan")
    assert weights["_native_plan_paths"]["sana_wm_vae_encoder_plan"].endswith(
        "sana_wm_vae_encoder_plan.plan"
    )
    assert weights["_tokenizer_sections"]["tokenizer.json"].endswith("text_encoder/tokenizer.json")

    extras = sana_wm_mod.plugin.build_extra_engines(cfg, weights, 256)
    for section, data in plans.items():
        assert extras[section] == data
    assert extras["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    assert extras["tokenizer_config.json"] == b'{"add_bos_token": true}'
    assert sana_wm_mod.plugin.build_engine(
        cfg, weights, 256
    ) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"

    overrides = sana_wm_mod.plugin.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert overrides["sana_wm_native_plan_sections"] == list(plans)


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

    cfg = ModelConfig.from_dir(tmp_path)
    weights = sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)

    assert "text_encoder_0_plan" not in weights["_native_plan_paths"]
    assert weights["_stage1_text_encoder_dir"].endswith("text_encoder")
    assert sana_wm_mod.plugin.build_engine(
        cfg,
        weights,
        512,
        precision="bf16",
    ) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"

    extras = sana_wm_mod.plugin.build_extra_engines(
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
    assert captured["max_cache_length"] == 512
    assert captured["precision"] == "bf16"
    assert captured["verbose"] is True
    overrides = sana_wm_mod.plugin.get_bundle_config_overrides(cfg)
    assert "engine_backend" not in overrides
    assert overrides["sana_wm_native_plan_sections"][0] == "text_encoder_0_plan"
    assert set(overrides["sana_wm_native_plan_sections"]) == set(plans) | {
        "text_encoder_0_plan"
    }


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
    weights = sana_wm_mod.plugin.load_weights(str(model_dir), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"] == str(
        plan_dir / "denoiser_plan.plan"
    )
    assert sana_wm_mod.plugin.build_engine(
        cfg, weights, 256
    ) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    assert sana_wm_mod.plugin.build_extra_engines(cfg, weights, 256)["denoiser_plan"] == plans[
        "denoiser_plan"
    ]


def test_sana_wm_plugin_discovers_native_plan_dir_from_env(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    plan_dir = tmp_path / "env_native_plans"
    _write_native_plan_set(model_dir, plan_dir=plan_dir)
    _write_tokenizer(model_dir)
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    monkeypatch.setenv("SANA_WM_NATIVE_PLAN_DIR", str(plan_dir))

    cfg = ModelConfig.from_dir(model_dir)
    weights = sana_wm_mod.plugin.load_weights(str(model_dir), cfg)

    assert weights["_native_plan_paths"]["denoiser_plan"] == str(
        plan_dir / "denoiser_plan.plan"
    )
    assert sana_wm_mod.plugin.build_engine(
        cfg, weights, 256
    ) == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"


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
    weights = sana_wm_mod.plugin.load_weights(str(model_dir), cfg)

    native_plan_paths = weights["_native_plan_paths"]
    assert native_plan_paths["denoiser_plan"] == str(
        external_model_dir / "trtmc_engines" / "denoiser_plan.plan"
    )
    assert sana_wm_mod.plugin.build_extra_engines(cfg, weights, 256)["denoiser_plan"] == plans[
        "denoiser_plan"
    ]


def test_sana_wm_plugin_rejects_partial_native_plan_sections(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    engine_dir = tmp_path / "trtmc_engines"
    engine_dir.mkdir()
    (engine_dir / "denoiser_plan.plan").write_bytes(b"stage1-dit-plan")

    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="complete prebuilt plan set"):
        sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)


def test_sana_wm_plugin_rejects_native_plan_sections_without_tokenizer(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    _write_native_plan_set(tmp_path)

    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="require tokenizer assets"):
        sana_wm_mod.plugin.load_weights(str(tmp_path), cfg)


def test_sana_wm_build_bundle_rejects_missing_native_sections(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text(_sana_yaml(), encoding="utf-8")
    output_path = str(tmp_path / "sana-wm.trtfb")

    monkeypatch.setattr(engine_builder, "_setup_trt_import", lambda rtx=False: None)
    monkeypatch.setattr(
        engine_builder.trt_compat, "resolved_summary", lambda: "mock TensorRT"
    )
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.0.0")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "mock-gpu")

    with pytest.raises(NotImplementedError, match="pure C\\+\\+ builds require native"):
        engine_builder.build_bundle(
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
    assert sections["engine_plan"] == b"TRTMC_SANA_WM_NATIVE_COMPONENTS\n"
    for section, data in plans.items():
        assert sections[section] == data
    assert sections["tokenizer.json"] == b'{"model": {"type": "Unigram"}}'
    config = json.loads(sections["config.json"].decode("utf-8"))
    assert config["engine_backend"] == "trt"
    assert "sana_wm_allow_python_bridge" not in config
    assert config["sana_wm_native_plan_sections"] == list(plans)
