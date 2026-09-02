# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3.ref2va_contract import (
    PresentationPiece,
    qwen_mrope_position_ids,
)
from tensorrt_model_connect.families.minimax_h3.ref2va_qwen_contract import (
    REF2VA_MAX_COMPACT_VISION_ROWS,
    REF2VA_MAX_VISION_PATCHES_PER_CALL,
    REF2VA_SHARED_TEXT_PROFILE,
    REF2VA_SHARED_VISION_PROFILE,
    ref2va_qwen_vision_invocations,
    ref2va_shared_qwen_abis,
)


def test_ref2va_shared_qwen_profiles_cover_public_endpoint_image_and_context() -> None:
    assert REF2VA_MAX_VISION_PATCHES_PER_CALL == 65_536
    assert REF2VA_MAX_COMPACT_VISION_ROWS == 262_144
    assert REF2VA_SHARED_VISION_PROFILE.max_patches == 65_536
    assert REF2VA_SHARED_VISION_PROFILE.min_patches == 2_040
    assert REF2VA_SHARED_TEXT_PROFILE.max_sequence_length == 262_144
    assert REF2VA_SHARED_TEXT_PROFILE.max_vision_rows == 262_144
    vision_abi, text_abi = ref2va_shared_qwen_abis()
    assert vision_abi.filename == "vision_encoder.plan"
    assert vision_abi.inputs[0].max_shape == (65_536, 1_536)
    assert vision_abi.inputs[0].min_shape == (2_040, 1_536)
    assert vision_abi.outputs[0].max_shape == (16_384, 5_120)
    assert text_abi.filename == "text_encoder.plan"
    assert text_abi.inputs[0].max_shape == (262_144,)
    assert text_abi.inputs[5].max_shape == (262_144, 5_120)


def test_qwen_vision_calls_do_not_spatially_chunk_and_keep_video_block_order() -> None:
    pieces = (
        PresentationPiece("text", "<Picture 1>: "),
        PresentationPiece("image", height=2_048, width=8_192),
        PresentationPiece("text", "<0.2 seconds>"),
        PresentationPiece("video", height=512, width=2_016),
        PresentationPiece("text", "<1.2 seconds>"),
        PresentationPiece("video", height=512, width=2_016),
    )
    calls = ref2va_qwen_vision_invocations(pieces)
    assert [call.modality for call in calls] == ["image", "video", "video"]
    assert [call.patch_rows for call in calls] == [65_536, 4_032, 4_032]
    assert [call.merged_rows for call in calls] == [16_384, 1_008, 1_008]
    assert all(call.source_frames == 2 for call in calls)
    with pytest.raises(ValueError, match="outside the shared plan profile"):
        ref2va_qwen_vision_invocations([PresentationPiece("image", height=32, width=32)])


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
