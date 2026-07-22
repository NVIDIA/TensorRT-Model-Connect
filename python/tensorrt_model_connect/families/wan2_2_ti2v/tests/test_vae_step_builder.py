# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the native recurrent Wan2.2 VAE decoder."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
    VAE_STEP_CACHE_SPECS,
    Wan22VaeStepProfile,
)


def test_vae_step_cache_contract_matches_source_order() -> None:
    profile = Wan22VaeStepProfile(44, 80)
    assert len(VAE_STEP_CACHE_SPECS) == 32
    assert VAE_STEP_CACHE_SPECS[0].logical_name == "decoder.conv_in"
    assert VAE_STEP_CACHE_SPECS[11].logical_name == ("decoder.up_blocks.0.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[18].logical_name == ("decoder.up_blocks.1.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[-1].logical_name == "decoder.conv_out"
    assert all(spec.shape(profile)[2] == 2 for spec in VAE_STEP_CACHE_SPECS)


@pytest.mark.parametrize(("height", "width"), [(0, 2), (2, 0), (-1, 2)])
def test_vae_step_profile_rejects_nonpositive_dimensions(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Wan22VaeStepProfile(height, width)
