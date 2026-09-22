# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from families.minimax_h3.config import SOL_ENGINE_1344X768_124F
from families.minimax_h3.vsa import make_vsa_geometry


def test_vsa_geometry_matches_the_fasth3_64_token_contract() -> None:
    profile = SOL_ENGINE_1344X768_124F
    geometry = make_vsa_geometry(profile)

    assert geometry.tile_shape == (4, 4, 4)
    assert geometry.num_prefix_tiles == 16
    assert geometry.num_video_tiles == 630
    assert geometry.num_tiles == 646
    assert geometry.topk_video_tiles == 63
    assert geometry.padded_rows == 646 * 64
    assert geometry.gather_indices.shape == (geometry.padded_rows,)
    assert geometry.untile_indices.shape == (profile.sequence_length,)
    assert geometry.variable_block_sizes.shape == (geometry.num_tiles,)
    assert geometry.variable_block_sizes.sum() == profile.sequence_length
    assert geometry.variable_block_sizes.min() > 0
    assert geometry.variable_block_sizes.max() == 64


def test_vsa_tile_and_untile_are_exact_inverses_for_live_rows() -> None:
    profile = SOL_ENGINE_1344X768_124F
    geometry = make_vsa_geometry(profile)
    source = np.arange(profile.sequence_length + 1, dtype=np.int32)
    source[-1] = -1

    tiled = source[geometry.gather_indices]

    np.testing.assert_array_equal(tiled[geometry.untile_indices], source[:-1])
    for tile, valid in enumerate(geometry.variable_block_sizes):
        begin = tile * geometry.tile_size
        assert np.all(tiled[begin + valid : begin + geometry.tile_size] == -1)


def test_vsa_video_tiles_follow_fastvideo_t_h_w_order() -> None:
    profile = SOL_ENGINE_1344X768_124F
    geometry = make_vsa_geometry(profile)
    first_video_slot = geometry.num_prefix_tiles * geometry.tile_size
    video_start = profile.text_rows + profile.audio_rows

    # First (4,4,4) tile flattened in source T/H/W row-major order.
    expected = [
        video_start + t * 28 * 36 + h * 36 + w
        for t in range(4)
        for h in range(4)
        for w in range(4)
    ]
    assert geometry.gather_indices[first_video_slot : first_video_slot + 64].tolist() == expected
    # The final video tile covers 1x4x4 because T=37, H=28, W=36.
    assert geometry.variable_block_sizes[-1] == 16
