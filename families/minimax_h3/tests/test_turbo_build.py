# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turbo is opt-in and must not alter the original delivery contract."""

import pytest

from families.minimax_h3 import model, staged_build
from families.minimax_h3.runtime_config_schema import normalize_build_options


def test_turbo_build_profile_is_distinct_but_prompt_stays_dynamic():
    original = model._public_dynamic_profile({})
    turbo = model._public_dynamic_profile({"turbo": True})
    assert original.video_rows == 108576
    assert turbo.video_rows == 113796
    assert turbo.audio_rows == 1206
    assert turbo.text_row_profile == original.text_row_profile == (1, 128, 2641)
    assert turbo.sequence_length == 117643
    assert turbo.first_block_cache  # Split-plan ABI, not a runtime cache policy.


def test_author_canvas_and_frames_require_turbo():
    request = {"height": 736, "width": 1280, "num_frames": 362}
    with pytest.raises(ValueError):
        model._default_canvas_size(request)
    with pytest.raises(ValueError):
        model._default_num_frames(request)
    request["turbo"] = True
    assert model._default_canvas_size(request) == (736, 1280)
    assert model._default_num_frames(request) == 362


@pytest.mark.parametrize(
    "conflict",
    [
        {"first_block_cache": True},
        {"first_block_cache_threshold": 0.3},
        {"quantized_transformer": "base.safetensors"},
        {"quantized_text_encoder": "text.safetensors"},
        {"super_resolution": True},
    ],
)
def test_turbo_rejects_silent_mixed_modes(conflict):
    with pytest.raises(ValueError):
        normalize_build_options({"turbo": True, **conflict})


def test_checkpoint_overrides_need_explicit_turbo():
    with pytest.raises(ValueError, match="require turbo=true"):
        normalize_build_options({"turbo_lora": "adapter.safetensors"})
    assert normalize_build_options({"turbo": True, "first_block_cache": False}) == {
        "turbo": True,
        "first_block_cache": False,
    }


def test_runtime_metadata_never_enables_turbo_cache_or_unauthenticated_ref_workflow():
    arguments = dict(
        trt_version="1.6.1",
        trt_abi="1_6",
        audio_vae_config={
            "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
            "sampling_rate": 32000,
            "latents_mean": [0.0] * 32,
            "latents_std": [1.0] * 32,
        },
        components=staged_build._COMPONENTS,
    )
    original = staged_build._runtime_config(**arguments)
    turbo = staged_build._runtime_config(**arguments, turbo_identity={"lora_strength": 1.0})
    assert "sampler" not in original
    assert original["scheduler_grid_points"] == 50
    assert original["transformer_forwards"] == 49
    assert original["first_block_cache"] is True
    assert original["num_frames_max"] == 345
    assert turbo["sampler"] == "turbo_euler"
    assert turbo["scheduler_grid_points"] == 9
    assert turbo["transformer_forwards"] == turbo["num_inference_steps"] == 8
    assert turbo["first_block_cache"] is False
    assert turbo["public_workflows"] == ["t2va", "fl2va"]
    assert "ref2va_transformer_ref" not in turbo
    assert turbo["num_frames_max"] == 362
    assert turbo["audio_latent_frames_max"] == 603
    assert turbo["text_rows_max"] == original["text_rows_max"] == 2641
