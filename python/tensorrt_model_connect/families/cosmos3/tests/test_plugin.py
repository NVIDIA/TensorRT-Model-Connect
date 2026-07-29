# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano model-owned plugin contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.families import find_diffusion_plugin
from tensorrt_model_connect.families.cosmos3.plugin import plugin
from tensorrt_model_connect.parallel_config import ParallelConfig


def test_matches_only_cosmos3_aliases() -> None:
    for alias in ("cosmos3", "Cosmos3-Nano", "cosmos3_nano"):
        assert plugin.matches(alias)
    assert not plugin.matches("cosmos_predict")
    assert not plugin.matches("wan_t2v")


def test_official_diffusers_pipeline_class_selects_plugin() -> None:
    assert find_diffusion_plugin("Cosmos3OmniDiffusersPipeline") is plugin


def test_load_weights_validates_checkpoint_layout(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    (root / "transformer").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "text_tokenizer").mkdir()
    (root / "assets").mkdir()
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    (root / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "latent_channel": 48,
                "latent_patch_size": 2,
                "patch_latent_dim": 192,
                "rms_norm_eps": 1.0e-6,
                "rope_theta": 5_000_000.0,
                "timestep_scale": 0.001,
                "hidden_act": "silu",
                "qk_norm_for_text": True,
                "rope_axes_dim": [24, 20, 20],
            }
        ),
        encoding="utf-8",
    )
    (root / "vae" / "config.json").write_text(
        json.dumps(
            {
                "z_dim": 48,
                "scale_factor_spatial": 16,
                "scale_factor_temporal": 4,
                "patch_size": 2,
            }
        ),
        encoding="utf-8",
    )
    (root / "text_tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "text_tokenizer" / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "assets" / "negative_prompt.json").write_text("{}", encoding="utf-8")

    weights = plugin.load_weights(str(root), SimpleNamespace(raw={}))
    assert weights["_model_format"] == "diffusers"
    assert weights["_transformer_config"]["num_hidden_layers"] == 36
    assert weights["_tokenizer_dir"].endswith("text_tokenizer")


def test_diffusion_config_uses_official_recipe_and_seed() -> None:
    cfg = plugin.get_diffusion_config(SimpleNamespace(raw={"seed": 42}))
    assert cfg == {
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "flow_shift": 10.0,
        "video_height": 720,
        "video_width": 1280,
        "video_num_frames": 189,
        "frame_rate": 24,
        "text_seq_len": 4096,
        "seed": 42,
    }
    with pytest.raises(ValueError, match="seed"):
        plugin.get_diffusion_config(SimpleNamespace(raw={"seed": -1}))


def test_context_parallel_bundle_uses_shared_rank_dynamic_plan_section() -> None:
    components = {
        "denoiser": b"cp-plan",
        "vae_decoder": b"vae",
        "vae_decoder_first_frame": b"vae-first",
        "tokenizer_json": b"{}",
        "tokenizer_config_json": b"{}",
    }
    sections = plugin.diffusion_bundle_sections(
        components,
        parallel_config=ParallelConfig(mode="context_parallel", cp_size=8),
    )
    assert sections[0] == ("denoiser_plan_cp", b"cp-plan")
