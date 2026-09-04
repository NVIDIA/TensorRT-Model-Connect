# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Superset shared-Qwen build contract required by MiniMax-H3 Ref2VA.

The existing multimodal vision and text builders own the only serialized copy
of Qwen3-VL.  Ref2VA does not introduce another Qwen plan.  It does, however,
require those shared plans to be rebuilt with the profiles below; the smaller
FL2VA-only defaults are not a valid public Ref2VA bundle.

This module is build/test tooling.  At inference the same rules are enforced by
the native C++ request packer and by bundle metadata.
"""

from __future__ import annotations

from .fl2va_contract import (
    MultimodalTextProfile,
    VisionEncoderProfile,
)
from .ref2va_contract import (
    QWEN_MAX_POSITION_EMBEDDINGS,
    QWEN_VISION_PATCH_SIZE,
)


# A public endpoint image resolves to 2048x8192.  Qwen patch embedding consumes
# one 2x16x16 RGB patch per row, so the image must remain one 65,536-row
# attention segment.  Spatial chunking would change Qwen attention and is
# forbidden.  A 768p video temporal block is much smaller (at most 4,176 rows).
REF2VA_MAX_VISION_PATCHES_PER_CALL = (2_048 // QWEN_VISION_PATCH_SIZE) * (
    8_192 // QWEN_VISION_PATCH_SIZE
)

# Compact visual rows are part of the language sequence.  The runtime validates
# the actual tokenizer-dependent presentation length <=262,144, so the shared
# plan must accept any compact-row count up to that same architectural limit.
REF2VA_MAX_COMPACT_VISION_ROWS = QWEN_MAX_POSITION_EMBEDDINGS

REF2VA_SHARED_VISION_PROFILE = VisionEncoderProfile(
    min_patches=2_040,
    opt_patches=4_032,
    max_patches=REF2VA_MAX_VISION_PATCHES_PER_CALL,
)
REF2VA_SHARED_TEXT_PROFILE = MultimodalTextProfile(
    min_sequence_length=1,
    opt_sequence_length=1_144,
    max_sequence_length=QWEN_MAX_POSITION_EMBEDDINGS,
    min_vision_rows=1,
    opt_vision_rows=1_008,
    max_vision_rows=REF2VA_MAX_COMPACT_VISION_ROWS,
)


def ref2va_shared_qwen_profile_metadata() -> dict[str, object]:
    """Path-free profile and invocation rules consumed by bundle validation."""

    vision = REF2VA_SHARED_VISION_PROFILE
    text = REF2VA_SHARED_TEXT_PROFILE
    return {
        "vision_encoder_plan": {
            "patch_rows_per_call": [
                vision.min_patches,
                vision.opt_patches,
                vision.max_patches,
            ],
            "invocation_unit": "one_image_or_one_two_frame_video_temporal_block",
            "spatial_chunking_allowed": False,
            "concatenate_outputs_in_reference_timestamp_order": True,
        },
        "text_encoder_plan": {
            "sequence_rows": [
                text.min_sequence_length,
                text.opt_sequence_length,
                text.max_sequence_length,
            ],
            "compact_vision_rows": [
                text.min_vision_rows,
                text.opt_vision_rows,
                text.max_vision_rows,
            ],
            "sequence_chunking_allowed": False,
        },
    }
