# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.families.minimax_h3.ref2va_contract import (
    qwen_mrope_position_ids,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_qwen_contract import (
    REF2VA_MAX_COMPACT_VISION_ROWS,
    REF2VA_MAX_VISION_PATCHES_PER_CALL,
    REF2VA_SHARED_TEXT_PROFILE,
    REF2VA_SHARED_VISION_PROFILE,
)


def test_ref2va_shared_qwen_profiles_cover_public_endpoint_image_and_context() -> None:
    assert REF2VA_MAX_VISION_PATCHES_PER_CALL == 65_536
    assert REF2VA_MAX_COMPACT_VISION_ROWS == 262_144
    assert REF2VA_SHARED_VISION_PROFILE.max_patches == 65_536
    assert REF2VA_SHARED_VISION_PROFILE.min_patches == 2_040
    assert REF2VA_SHARED_TEXT_PROFILE.max_sequence_length == 262_144
    assert REF2VA_SHARED_TEXT_PROFILE.max_vision_rows == 262_144


def test_qwen_video_grid_repeat_interleave_matches_expanded_temporal_calls() -> None:
    # Two independent four-pad video runs separated by timestamp text.
    token_types = (0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0)
    released_grid = qwen_mrope_position_ids(
        token_types,
        image_grids=(),
        video_grids=((2, 4, 4),),
    )
    expanded_call_grids = qwen_mrope_position_ids(
        token_types,
        image_grids=(),
        video_grids=((1, 4, 4), (1, 4, 4)),
    )
    assert np.array_equal(released_grid, expanded_call_grids)
