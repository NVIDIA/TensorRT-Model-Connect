# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-gate tests for the native Wan2.2 qualification tool."""

from __future__ import annotations

from tensorrt_model_connect.families.wan2_2_ti2v.reference.compare_native_pngs import (
    DEFAULT_MINIMUM_COSINE,
    _accuracy_failures,
)


def test_accuracy_gate_accepts_the_declared_boundary() -> None:
    assert DEFAULT_MINIMUM_COSINE == 0.998
    assert not _accuracy_failures(
        cosine=DEFAULT_MINIMUM_COSINE,
        minimum_frame_cosine=DEFAULT_MINIMUM_COSINE,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
    )


def test_accuracy_gate_rejects_video_or_frame_regressions() -> None:
    assert _accuracy_failures(
        cosine=0.997,
        minimum_frame_cosine=0.996,
        min_cosine=DEFAULT_MINIMUM_COSINE,
        min_frame_cosine=DEFAULT_MINIMUM_COSINE,
    ) == [
        "cosine_uint8=0.997000000000 < 0.998000000000",
        "minimum_frame_cosine_uint8=0.996000000000 < 0.998000000000",
    ]
