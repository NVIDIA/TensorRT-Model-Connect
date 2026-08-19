# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned package smoke defaults."""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("package_smoke.json")


def test_qwen_package_smoke_config_matches_current_ci_default() -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert data["default"] is True
    assert data["name"] == "qwen3-0.6b"
    assert data["model_id"] == "Qwen/Qwen3-0.6B"
    assert data["bundle"] == "qwen3-0.6b.bundle"
    assert data["timing_cache"] == "qwen3-0.6b.timing.cache"
    assert data["max_cache"] == 64
    assert data["max_new_tokens"] == 8
    assert data["optimization_level"] == 1
    assert data["build_timeout"] == "45m"
    assert data["run_timeout"] == "10m"
    assert data["precision"] == "fp16"
    assert data["prompt"] == "The capital of France is"
    assert data["run_args"] == ["--greedy"]
