# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the native recurrent Wan2.2 VAE decoder."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
    OFFICIAL_VAE_STEP_PROFILE,
    SMALL_VAE_STEP_PROFILE,
    VAE_STEP_CACHE_SPECS,
    Wan22VaeStepProfile,
    _configure_step_builder_config,
    vae_step_cache_bytes,
)


def test_vae_step_profiles() -> None:
    assert SMALL_VAE_STEP_PROFILE.latent_shape == (1, 48, 1, 2, 2)
    assert SMALL_VAE_STEP_PROFILE.video_shape(first_frame_only=True) == (
        1,
        3,
        1,
        32,
        32,
    )
    assert SMALL_VAE_STEP_PROFILE.video_shape(first_frame_only=False) == (
        1,
        3,
        4,
        32,
        32,
    )
    assert OFFICIAL_VAE_STEP_PROFILE.latent_shape == (1, 48, 1, 44, 80)
    assert OFFICIAL_VAE_STEP_PROFILE.video_shape(first_frame_only=True) == (
        1,
        3,
        1,
        704,
        1280,
    )
    assert OFFICIAL_VAE_STEP_PROFILE.video_shape(first_frame_only=False) == (
        1,
        3,
        4,
        704,
        1280,
    )


def test_vae_step_cache_contract_matches_source_order() -> None:
    assert len(VAE_STEP_CACHE_SPECS) == 32
    assert [spec.index for spec in VAE_STEP_CACHE_SPECS] == list(range(32))
    assert VAE_STEP_CACHE_SPECS[0].logical_name == "decoder.conv_in"
    assert VAE_STEP_CACHE_SPECS[11].logical_name == ("decoder.up_blocks.0.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[18].logical_name == ("decoder.up_blocks.1.upsampler.time_conv")
    assert VAE_STEP_CACHE_SPECS[-1].logical_name == "decoder.conv_out"
    assert all(spec.shape(OFFICIAL_VAE_STEP_PROFILE)[2] == 2 for spec in VAE_STEP_CACHE_SPECS)


def test_vae_step_cache_footprint() -> None:
    assert vae_step_cache_bytes(SMALL_VAE_STEP_PROFILE) == 7_308_800
    assert vae_step_cache_bytes(OFFICIAL_VAE_STEP_PROFILE) == 6_431_744_000


class _FakeMemoryPoolType:
    WORKSPACE = "workspace"


class _FakeTrt:
    MemoryPoolType = _FakeMemoryPoolType


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.pool_limits: list[tuple[str, int]] = []
        self.max_aux_streams = 17

    def set_memory_pool_limit(self, pool: str, size: int) -> None:
        self.pool_limits.append((pool, size))


def test_vae_step_builder_can_force_single_stream_without_changing_default() -> None:
    default_config = _FakeBuilderConfig()
    _configure_step_builder_config(_FakeTrt, default_config, workspace_gib=64, max_aux_streams=None)
    assert default_config.pool_limits == [("workspace", 64 << 30)]
    assert default_config.max_aux_streams == 17

    single_stream_config = _FakeBuilderConfig()
    _configure_step_builder_config(
        _FakeTrt, single_stream_config, workspace_gib=64, max_aux_streams=0
    )
    assert single_stream_config.pool_limits == [("workspace", 64 << 30)]
    assert single_stream_config.max_aux_streams == 0


@pytest.mark.parametrize(
    ("workspace_gib", "max_aux_streams", "message"),
    [(0, None, "workspace_gib"), (64, -1, "max_aux_streams")],
)
def test_vae_step_builder_rejects_invalid_resource_limits(
    workspace_gib: int, max_aux_streams: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _configure_step_builder_config(
            _FakeTrt,
            _FakeBuilderConfig(),
            workspace_gib=workspace_gib,
            max_aux_streams=max_aux_streams,
        )


@pytest.mark.parametrize(("height", "width"), [(0, 2), (2, 0), (-1, 2)])
def test_vae_step_profile_rejects_nonpositive_dimensions(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Wan22VaeStepProfile(height, width)
