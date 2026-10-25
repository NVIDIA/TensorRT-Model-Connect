# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the native gather mapping with H3's scatter-style reference math."""

import numpy as np
import pytest

from families.minimax_h3.packed_rows import PackedRows


def test_interleaved_refs_preserve_original_operand_order_and_target_suffixes():
    # text | image | video soundtrack | video frames | standalone audio |
    # generated audio | generated video; each modality operand keeps that order.
    rows = PackedRows(
        np.array([0, 1, 2]), np.array([5, 6, 10, 11, 12]), np.array([3, 4, 7, 8, 9, 13, 14, 15, 16])
    )
    rng = np.random.default_rng(41)
    text = rng.normal(size=(3, 5)).astype(np.float32)
    audio = rng.normal(size=(5, 5)).astype(np.float32)
    video = rng.normal(size=(9, 5)).astype(np.float32)
    original = np.zeros((17, 5), dtype=np.float32)
    original[rows.text_indices] = text
    original[rows.video_indices] = video
    original[rows.audio_indices] = audio
    actual = np.concatenate((text, audio, video))[rows.packed_to_concat]
    np.testing.assert_array_equal(actual, original)
    generated_video, generated_audio = rows.generated_indices(
        condition_video_rows=5, condition_audio_rows=3
    )
    np.testing.assert_array_equal(generated_video, [13, 14, 15, 16])
    np.testing.assert_array_equal(generated_audio, [11, 12])


def test_existing_text_audio_video_layout_remains_an_identity_permutation():
    rows = PackedRows(np.arange(3), np.arange(3, 5), np.arange(5, 9))
    np.testing.assert_array_equal(rows.packed_to_concat, np.arange(9))


def test_interleaved_operand_indices_do_not_need_to_be_sorted():
    rows = PackedRows(np.array([2, 0]), np.array([4]), np.array([3, 1]))
    np.testing.assert_array_equal(rows.packed_to_concat, [1, 4, 0, 3, 2])


def test_index_arrays_are_owned_and_read_only():
    source = np.array([0, 1], dtype=np.int64)
    rows = PackedRows(source, np.array([2]), np.array([3]))
    source[:] = 99
    np.testing.assert_array_equal(rows.text_indices, [0, 1])
    assert rows.text_indices.dtype == np.int32
    with pytest.raises(ValueError):
        rows.text_indices[0] = 99


@pytest.mark.parametrize(
    "text,audio,video",
    [
        ([0, 0], [1], [2]),
        ([0], [2], [3]),
        ([-1], [1], [2]),
        ([0.0], [1], [2]),
        ([[0]], [1], [2]),
        ([False], [1], [2]),
        ([2**32], [1], [2]),
    ],
)
def test_rejects_invalid_or_nonexhaustive_layout(text, audio, video):
    with pytest.raises(ValueError):
        PackedRows(np.asarray(text), np.asarray(audio), np.asarray(video))


@pytest.mark.parametrize("video,audio", [(-1, 0), (3, 0), (0, 2), (True, 0), (1.5, 0)])
def test_rejects_invalid_condition_prefix_sizes(video, audio):
    rows = PackedRows(np.array([0]), np.array([1]), np.array([2, 3]))
    with pytest.raises(ValueError):
        rows.generated_indices(condition_video_rows=video, condition_audio_rows=audio)
