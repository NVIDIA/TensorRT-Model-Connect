# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest

from families.minimax_h3.config import (
    SOL_ENGINE_1344X768_124F,
    SOL_ENGINE_1344X768_124_TO_345F,
)


def test_public_profile_covers_dynamic_text_and_full_media_envelope() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F
    assert profile.text_row_profile == (1, 128, 2641)
    assert profile.video_row_profile == (14985, 37296, 108576)
    assert profile.audio_row_profile == (414, 414, 1150)
    assert profile.packed_row_profile == (15400, 37838, 112367)
    profile.validate()


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
