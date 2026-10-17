# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact row permutations for H3's interleaved multimodal reference sequence.

Projection/refinement operates separately on text, audio and video operands.
Ref2VA then interleaves those operands in reference order. These permutations
are the boundary between that layout and the family-owned TensorRT graph;
they do not infer a modality tag or noise level from an operand's projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedRows:
    """An exhaustive disjoint mapping between modality operands and sequence."""

    text_indices: np.ndarray
    audio_indices: np.ndarray
    video_indices: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for name in ("text_indices", "audio_indices", "video_indices"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"H3 {name} must be a one-dimensional integer array")
            if value.size > np.iinfo(np.int32).max:
                raise ValueError("H3 row count exceeds the TensorRT int32 index range")
            if value.size and (value.min() < 0 or value.max() > np.iinfo(np.int32).max):
                raise ValueError(f"H3 {name} contains an invalid packed row index")
            owned = np.array(value, dtype=np.int32, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
            arrays.append(owned)
        indices = np.concatenate(arrays)
        if indices.size == 0 or indices.size > np.iinfo(np.int32).max:
            raise ValueError("H3 requires a nonempty int32-addressable packed sequence")
        if not np.array_equal(np.sort(indices), np.arange(indices.size, dtype=np.int32)):
            raise ValueError("H3 modality indices must partition every packed row exactly once")

    @property
    def row_count(self) -> int:
        return self.text_indices.size + self.audio_indices.size + self.video_indices.size

    @property
    def packed_to_concat(self) -> np.ndarray:
        """Gather rows of concat(text, audio, video) into the packed sequence."""

        concatenated_positions = np.concatenate(
            (self.text_indices, self.audio_indices, self.video_indices)
        )
        inverse = np.empty(self.row_count, dtype=np.int32)
        inverse[concatenated_positions] = np.arange(self.row_count, dtype=np.int32)
        return inverse

    def generated_indices(self, *, condition_video_rows: int, condition_audio_rows: int):
        """Return target rows only; reference rows lead each modality operand."""

        for count, available in (
            (condition_video_rows, self.video_indices.size),
            (condition_audio_rows, self.audio_indices.size),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= available:
                raise ValueError("H3 condition row count must be an integer within its operand")
        return (
            self.video_indices[condition_video_rows:],
            self.audio_indices[condition_audio_rows:],
        )
