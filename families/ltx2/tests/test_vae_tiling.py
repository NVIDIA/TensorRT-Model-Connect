# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile plan of the tiled / tile-parallel video VAE decode (no TensorRT, torch or GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest

from families.ltx2.vae_tiling import TileConfig, assign_lpt, axis_weights, blend_tiles, plan_tiles, split_axis

GRIDS = [(31, 22, 40), (16, 17, 30), (31, 11, 20), (5, 6, 7), (3, 2, 3), (40, 9, 9)]
CONFIGS = [TileConfig(), TileConfig(tile_frames=0), TileConfig(tile_frames=136, overlap_frames=16),
           TileConfig(tile_pixels=192, overlap_pixels=64, tile_frames=40, overlap_frames=8),
           TileConfig(tile_frames=96, overlap_frames=24)]


def _weight_sum(plan: dict, frames: int, height: int, width: int) -> np.ndarray:
    tt, th, tw = plan["tile_pixels"]
    den = np.zeros((frames, height, width), np.float64)
    for tile in plan["tiles"]:
        t0, y0, x0 = tile["pixel_start"]
        (tl, tr), (yl, yr), (xl, xr) = tile["ramps"]
        w = (axis_weights(tt, tl, tr, temporal=True)[:, None, None]
             * axis_weights(th, yl, yr, temporal=False)[None, :, None]
             * axis_weights(tw, xl, xr, temporal=False)[None, None, :])
        den[t0:t0 + tt, y0:y0 + th, x0:x0 + tw] += w
    return den


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("grid", GRIDS)
def test_tiles_cover_the_video_with_unit_weight_sums(grid, config) -> None:
    f, h, w = grid
    plan = plan_tiles(f, h, w, config, world=2)
    frames, height, width = (f - 1) * 8 + 1, h * 32, w * 32
    tf, th, tw = plan["tile_latent"]
    assert plan["tile_pixels"] == [(tf - 1) * 8 + 1, th * 32, tw * 32]
    for tile in plan["tiles"]:
        f0, h0, w0 = tile["latent_start"]
        assert 0 <= f0 and f0 + tf <= f and 0 <= h0 and h0 + th <= h and 0 <= w0 and w0 + tw <= w
        assert tile["pixel_start"] == [f0 * 8, h0 * 32, w0 * 32]
    np.testing.assert_allclose(_weight_sum(plan, frames, height, width), 1.0, atol=1e-6)


def test_default_plan_for_the_large_config() -> None:
    plan = plan_tiles(31, 22, 40, TileConfig(), world=2)
    # Spatial 512 px tiles with >= 64 px overlap (the diffusers enable_tiling geometry, equalized);
    # 241 frames fit one temporal tile.
    assert plan["tile_latent"] == [31, 12, 15]
    assert len(plan["tiles"]) == 6
    assert [t["rank"] for t in plan["tiles"]] == [0, 1, 0, 1, 0, 1]
    assert {t["latent_start"][2] for t in plan["tiles"]} == {0, 13, 25}
    assert {t["latent_start"][1] for t in plan["tiles"]} == {0, 10}


def test_split_overlaps_are_at_least_the_minimum() -> None:
    for length in range(1, 80):
        for tile, overlap in ((16, 2), (8, 2), (12, 3), (32, 4)):
            size, starts = split_axis(length, tile, overlap)
            assert size <= max(tile, length if length <= tile else 0) or len(starts) == 1
            assert starts[0] == 0 and starts[-1] + size == length
            overlaps = [starts[i] + size - starts[i + 1] for i in range(len(starts) - 1)]
            assert all(o >= overlap for o in overlaps)
            assert all(a + b <= size for a, b in zip([0, *overlaps], [*overlaps, 0]))


def test_temporal_ramps_follow_the_causal_frame_mapping() -> None:
    plan = plan_tiles(31, 4, 4, TileConfig(tile_pixels=0, tile_frames=136, overlap_frames=16))
    first, second = plan["tiles"]
    assert first["latent_start"][0] == 0 and second["latent_start"][0] == 14
    # Latent overlap 3 -> (3 - 1) * 8 + 1 = 17 shared frames: the later tile fades in from 0 over
    # all of them, the earlier one fades out over the last 16.
    assert second["ramps"][0] == [17, 0] and first["ramps"][0] == [0, 16]
    w = axis_weights(plan["tile_pixels"][0], 17, 0, temporal=True)
    assert w[0] == 0.0 and w[16] == np.float32(16) / np.float32(17) and w[17] == 1.0


def test_lpt_balances_unequal_volumes() -> None:
    ranks = assign_lpt([5, 4, 3, 3, 1], 2)
    load = [sum(v for v, r in zip([5, 4, 3, 3, 1], ranks) if r == k) for k in range(2)]
    assert sorted(load) == [8, 8]


def test_blend_of_constant_tiles_is_the_constant() -> None:
    plan = plan_tiles(5, 6, 7, TileConfig(tile_pixels=192, overlap_pixels=64, tile_frames=40, overlap_frames=8))
    tiles = [np.full(plan["tile_pixels"] + [3], 0.375, np.float16) for _ in plan["tiles"]]
    out = blend_tiles(plan, tiles, 33, 192, 224)
    np.testing.assert_allclose(out, 0.375, rtol=1e-6)


@pytest.mark.parametrize("config", [TileConfig(tile_pixels=48), TileConfig(overlap_pixels=512),
                                    TileConfig(overlap_pixels=256), TileConfig(tile_frames=20),
                                    TileConfig(tile_frames=64, overlap_frames=24), TileConfig(tile_pixels=-32)])
def test_invalid_tile_configs_are_rejected(config) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_family_build_command_declares_the_tile_options() -> None:
    from tensorrt_model_connect import family_cli

    declaration = family_cli.load_family_cli("ltx2")
    build = next(c for c in declaration["commands"] if c["name"] == "build")
    names = {a["name"] for a in build["arguments"]}
    assert {"vae_tile_pixels", "vae_tile_overlap_pixels", "vae_tile_frames", "vae_tile_overlap_frames"} <= names
    defaults = {a["name"]: a.get("default") for a in build["arguments"]}
    config = TileConfig()
    assert (defaults["vae_tile_pixels"], defaults["vae_tile_overlap_pixels"], defaults["vae_tile_frames"],
            defaults["vae_tile_overlap_frames"]) == (config.tile_pixels, config.overlap_pixels, config.tile_frames,
                                                     config.overlap_frames)
