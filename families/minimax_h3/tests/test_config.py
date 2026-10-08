# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from families.minimax_h3.config import (
    FASTH3_DENSE_4STEP_GENERATION_PROFILE,
    FASTH3_VSA_4STEP_GENERATION_PROFILE,
    SOL_ENGINE_1344X768_124F,
    MiniMaxH3GenerationProfile,
)


def test_dynamic_text_profile_preserves_the_537_token_maximum() -> None:
    profile = SOL_ENGINE_1344X768_124F
    assert (profile.min_text_rows, profile.opt_text_rows, profile.text_rows) == (1, 128, 537)
    assert (
        profile.min_sequence_length,
        profile.opt_sequence_length,
        profile.sequence_length,
    ) == (37711, 37838, 38247)
    assert profile.padded_sequence_length == profile.sequence_length
    profile.validate()


@pytest.mark.parametrize(
    "profile",
    (
        replace(SOL_ENGINE_1344X768_124F, min_text_rows=0),
        replace(SOL_ENGINE_1344X768_124F, min_text_rows=129, opt_text_rows=128),
        replace(SOL_ENGINE_1344X768_124F, opt_text_rows=538),
    ),
)
def test_dynamic_text_profile_rejects_invalid_bounds(profile) -> None:
    with pytest.raises(ValueError, match="1 <= min <= opt <= max"):
        profile.validate()


def test_fasth3_dense_generation_profile_is_four_transformer_forwards() -> None:
    profile = FASTH3_DENSE_4STEP_GENERATION_PROFILE
    profile.validate()
    assert profile.num_inference_steps == 5
    assert profile.transformer_forwards == 4


def test_fasth3_vsa_generation_profile_is_the_trained_sm100a_recipe() -> None:
    profile = FASTH3_VSA_4STEP_GENERATION_PROFILE
    profile.validate()
    assert profile.transformer_forwards == 4
    assert profile.uses_vsa
    assert (profile.vsa_tile_size, profile.vsa_sparsity, profile.vsa_kernel) == (
        64,
        0.9,
        "sm100a",
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("vsa_tile_size", 256, "64-token"),
        ("vsa_sparsity", 0.8, "sparsity=0.9"),
        ("vsa_kernel", "triton", "sm100a kernel contract"),
    ),
)
def test_fasth3_vsa_rejects_recipe_drift(field: str, value, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(FASTH3_VSA_4STEP_GENERATION_PROFILE, **{field: value}).validate()


@pytest.mark.parametrize(
    "rungs",
    (
        (999, 749, 500),
        (999, 749, 749, 250),
        (1000, 749, 500, 250),
        (999, 749, 500, 0),
    ),
)
def test_generation_profile_rejects_invalid_dmd_ladder(rungs: tuple[int, ...]) -> None:
    profile = MiniMaxH3GenerationProfile(
        name="invalid",
        num_inference_steps=5,
        video_scheduler_shift=12.0,
        audio_scheduler_shift=3.0,
        dmd_denoising_steps=rungs,
    )
    with pytest.raises(ValueError, match="DMD"):
        profile.validate()
