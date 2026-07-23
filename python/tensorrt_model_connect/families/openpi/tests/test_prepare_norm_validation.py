# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.families.openpi.prepare_model_dir import (
    _load_and_validate_norm_stats,
)


def _official_style_droid_stats() -> dict:
    active_low = [-1.0] * 8
    active_high = [1.0] * 8
    padded = [0.0] * 24
    return {
        "norm_stats": {
            "state": {
                "q01": active_low + padded,
                "q99": active_high + padded,
            },
            "actions": {
                "q01": active_low + padded,
                "q99": active_high + padded,
            },
        }
    }


def test_official_droid_zero_span_padding_is_accepted(tmp_path) -> None:
    stats = _official_style_droid_stats()
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    assert _load_and_validate_norm_stats(path, state_dim=8, action_dim=8) == stats


def test_decreasing_quantiles_in_unused_padding_are_rejected(tmp_path) -> None:
    stats = _official_style_droid_stats()
    stats["norm_stats"]["state"]["q99"][31] = -1.0
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match=r"state.*\[31\].*q99 < q01"):
        _load_and_validate_norm_stats(path, state_dim=8, action_dim=8)
