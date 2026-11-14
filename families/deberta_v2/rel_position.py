# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relative-position index matrices, built once as weights.

DeBERTa's disentangled attention needs, for every (query, key) pair, the
bucket of their distance. The reference recomputes that matrix on every
forward pass; the engine runs at one fixed length, so here it is a build-time
constant - the trade ``families/dinov2`` makes with its interpolated position
table and ``families/nomic_bert`` with its rotary tables.

The bucketing is logarithmic, not linear: distances below ``bucket_size / 2``
keep their own value, and everything beyond is compressed onto a log scale so
that one bucket covers a wider span the further away it is. Reproducing it
as a plain clamp would give an index matrix of the right shape with the wrong
values, which is the kind of error that runs.
"""

from __future__ import annotations

import numpy as np


def log_bucket(relative: np.ndarray, bucket_size: int, max_position: int) -> np.ndarray:
    """``make_log_bucket_position``, in integers.

    Computed in float64 and then rounded, because the reference takes a
    ``ceil`` of a ratio of logarithms and a float32 evaluation lands on the
    other side of an integer for some distances.
    """
    sign = np.sign(relative)
    mid = bucket_size // 2
    absolute = np.where((relative < mid) & (relative > -mid), mid - 1, np.abs(relative))
    absolute = absolute.astype(np.float64)
    log_pos = np.ceil(np.log(absolute / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket = np.where(np.abs(relative) <= mid, relative.astype(np.float64), log_pos * sign)
    return bucket.astype(np.int64)


def relative_matrix(seq_len: int, bucket_size: int, max_position: int) -> np.ndarray:
    """``[seq, seq]`` of bucketed ``query_position - key_position``."""
    positions = np.arange(seq_len, dtype=np.int64)
    relative = positions[:, None] - positions[None, :]
    if bucket_size > 0 and max_position > 0:
        relative = log_bucket(relative, bucket_size, max_position)
    return relative


def gather_indices(seq_len: int, bucket_size: int, max_position: int,
                   span: int) -> tuple[np.ndarray, np.ndarray]:
    """The two index matrices the c2p and p2c terms gather with.

    Both are ``clamp(+-relative + span, 0, 2 * span - 1)``. The sign is the
    whole difference between them: content-to-position looks the distance up
    forwards, position-to-content looks it up backwards.
    """
    relative = relative_matrix(seq_len, bucket_size, max_position)
    limit = 2 * span - 1
    content_to_position = np.clip(relative + span, 0, limit)
    position_to_content = np.clip(-relative + span, 0, limit)
    return content_to_position.astype(np.int32), position_to_content.astype(np.int32)
