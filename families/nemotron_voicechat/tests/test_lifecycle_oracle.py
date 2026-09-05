# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.nemotron_voicechat.tests.lifecycle_oracle import (
    REQUIRED_SECTIONS,
    assert_lifecycle_receipt,
)


def test_lifecycle_oracle_requires_every_active_section() -> None:
    assert REQUIRED_SECTIONS == (
        "baseline",
        "irregular_chunking",
        "barge_in",
        "cancel",
        "reset_vs_fresh",
        "processed_input_clear",
        "response_cancel_recovery",
        "response_truncate_recovery",
        "partial_finish_tail",
        "sequence_continuity",
        "media_continuity",
        "normal_multiturn",
        "function_channel",
        "backpressure_concurrency",
    )


def test_lifecycle_oracle_fails_closed_without_receipt_sections() -> None:
    with pytest.raises(AssertionError, match="missing section baseline"):
        assert_lifecycle_receipt(
            {
                "schema_version": 3,
                "runtime": "C++ ISpeechSession with TensorRT backend",
            },
            "expected",
        )
