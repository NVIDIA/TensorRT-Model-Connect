# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano architecture, generation, and CP layout contracts."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.cosmos3.model_config import (
    COSMOS3_NANO,
    context_parallel_layout,
    select_generation_profile,
    validate_transformer_config,
)


def _transformer_config() -> dict[str, object]:
    return {
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
        "use_und_k_norm_for_gen": False,
        "rope_axes_dim": [24, 20, 20],
    }


def test_official_profile_has_full_video_geometry() -> None:
    assert COSMOS3_NANO.video_width == 1280
    assert COSMOS3_NANO.video_height == 720
    assert COSMOS3_NANO.video_num_frames == 189
    assert COSMOS3_NANO.num_inference_steps == 35
    assert COSMOS3_NANO.latent_frames == 48
    assert COSMOS3_NANO.latent_height == 45
    assert COSMOS3_NANO.latent_width == 80
    assert COSMOS3_NANO.patch_height == 23
    assert COSMOS3_NANO.patch_width == 40
    assert COSMOS3_NANO.num_vision_tokens == 44_160
    assert COSMOS3_NANO.max_text_seq_len == 4096


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_cp_layout_supports_requested_degrees(world_size: int) -> None:
    layout = context_parallel_layout(197, COSMOS3_NANO.num_vision_tokens, world_size)
    assert layout.padded_text_tokens % world_size == 0
    assert layout.padded_vision_tokens % world_size == 0
    assert layout.local_vision_tokens * world_size == layout.padded_vision_tokens
    assert COSMOS3_NANO.num_attention_heads % world_size == 0


def test_cp_layout_pads_streams_independently() -> None:
    layout = context_parallel_layout(197, 44_161, 8)
    assert layout.padded_text_tokens == 200
    assert layout.padded_vision_tokens == 44_168
    assert layout.text_padding == 3
    assert layout.vision_padding == 7


def test_only_full_quality_profile_is_qualified() -> None:
    assert select_generation_profile({}) is COSMOS3_NANO
    with pytest.raises(ValueError, match="full T2V profile"):
        select_generation_profile({"video_num_frames": 9, "num_inference_steps": 2})


def test_transformer_config_is_exact_and_fail_closed() -> None:
    raw = _transformer_config()
    validate_transformer_config(raw)
    raw["num_hidden_layers"] = 64
    with pytest.raises(ValueError, match="num_hidden_layers=64"):
        validate_transformer_config(raw)
