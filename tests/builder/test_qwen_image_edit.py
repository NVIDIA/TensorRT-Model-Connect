"""Qwen-Image Edit build/config safety tests.

Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
Intent: lock the Qwen-Image-Edit-2511 detection path so it captures the
Edit-specific metadata and does not silently emit a text-to-image bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families import find_diffusion_plugin, find_plugin
from tensorrt_model_connect.families.qwen_image.qwen_image_bundle_config import (
    build_bundle_config,
)


def _write_stub_qwen_image_repo(tmp_path: Path, *, edit: bool) -> Path:
    repo = tmp_path / ("qwen-image-edit-2511" if edit else "qwen-image-2512")
    repo.mkdir()

    pipeline_class = "QwenImageEditPlusPipeline" if edit else "QwenImagePipeline"
    (repo / "model_index.json").write_text(
        json.dumps({"_class_name": pipeline_class}), encoding="utf-8")

    (repo / "transformer").mkdir()
    (repo / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "in_channels": 64,
                "out_channels": 16,
                "patch_size": 2,
                "num_layers": 60,
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "joint_attention_dim": 3584,
                "axes_dims_rope": [16, 56, 56],
                "guidance_embeds": False,
            }
        ),
        encoding="utf-8",
    )

    (repo / "vae").mkdir()
    (repo / "vae" / "config.json").write_text(
        json.dumps(
            {
                "z_dim": 16,
                "base_dim": 96,
                "dim_mult": [1, 2, 4, 4],
                "temperal_downsample": [False, True, True],
                "latents_mean": [0.0] * 16,
                "latents_std": [1.0] * 16,
            }
        ),
        encoding="utf-8",
    )

    (repo / "text_encoder").mkdir()
    (repo / "text_encoder" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_vl",
                "text_config": {
                    "hidden_size": 3584,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 28,
                    "num_key_value_heads": 4,
                    "intermediate_size": 18944,
                    "vocab_size": 152064,
                    "rope_theta": 1000000.0,
                    "rms_norm_eps": 1e-6,
                },
                "vision_config": {
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "hidden_size": 1280,
                    "depth": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    (repo / "scheduler").mkdir()
    (repo / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "use_dynamic_shifting": True,
                "base_shift": 0.5,
                "max_shift": 0.9,
                "base_image_seq_len": 256,
                "max_image_seq_len": 8192,
                "shift_terminal": 0.02,
                "time_shift_type": "exponential",
            }
        ),
        encoding="utf-8",
    )

    for name in ("tokenizer", "processor"):
        (repo / name).mkdir()

    return repo


def test_qwen_image_edit_bundle_config_captures_edit_metadata(tmp_path: Path) -> None:
    repo = _write_stub_qwen_image_repo(tmp_path, edit=True)

    cfg = build_bundle_config(repo)

    assert cfg["task_mode"] == "edit"
    assert cfg["text_encoder"]["type"] == "qwen2_5_vl_multimodal"
    assert cfg["tokenizer"]["prompt_template_kind"] == "qwen_image_edit_hardcoded"
    assert cfg["tokenizer"]["prompt_template_drop_idx"] == 64
    assert cfg["vae"]["has_encoder"] is True
    assert cfg["diffusion"]["use_dynamic_shifting"] is True
    assert cfg["diffusion"]["base_shift"] == 0.5
    assert cfg["diffusion"]["max_shift"] == 0.9
    assert cfg["diffusion"]["shift_terminal"] == 0.02
    assert cfg["diffusion"]["time_shift_type"] == "exponential"
    assert cfg["vision_encoder"]["type"] == "qwen2_5_vl_vision"
    assert cfg["vision_encoder"]["image_size"] == 384
    assert cfg["image_conditioning"]["vae_concat_axis"] == "sequence"
    assert cfg["image_conditioning"]["max_input_images"] == 1


def test_qwen_image_t2i_bundle_config_keeps_t2i_metadata(tmp_path: Path) -> None:
    repo = _write_stub_qwen_image_repo(tmp_path, edit=False)

    cfg = build_bundle_config(repo)

    assert cfg["task_mode"] == "t2i"
    assert cfg["text_encoder"]["type"] == "qwen2_5_vl_lm"
    assert cfg["tokenizer"]["prompt_template_kind"] == "qwen_image_t2i_hardcoded"
    assert cfg["tokenizer"]["prompt_template_drop_idx"] == 34
    assert cfg["vae"]["has_encoder"] is False
    assert "vision_encoder" not in cfg
    assert "image_conditioning" not in cfg


def test_qwen_image_plugin_claims_edit_pipeline_classes() -> None:
    plugin = find_diffusion_plugin("QwenImageEditPlusPipeline")
    assert plugin is not None
    assert plugin.name == "qwen_image"
    assert find_plugin("qwen_image_edit") is plugin


def test_qwen_image_edit_build_fails_before_t2i_engine_build(tmp_path: Path) -> None:
    repo = _write_stub_qwen_image_repo(tmp_path, edit=True)
    plugin = find_plugin("qwen_image")
    assert plugin is not None
    config = ModelConfig(
        model_type="qwen_image",
        raw=json.loads((repo / "model_index.json").read_text(encoding="utf-8")),
    )
    weights = plugin.load_weights(str(repo), config)

    with pytest.raises(NotImplementedError, match="Qwen-Image Edit"):
        plugin.build_components(str(repo), config, weights, precision="bf16")
