# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release-performance contract for Qwen3-4B generation."""

from pathlib import Path

import yaml


RELEASE_CONFIG = Path(__file__).resolve().parents[4] / "benchmarks/performance/release.yaml"


def test_qwen3_4b_uses_cuda_graphs_with_an_fp16_reference() -> None:
    release = yaml.safe_load(RELEASE_CONFIG.read_text(encoding="utf-8"))
    profile = next(
        item
        for item in release["additional_profiles"]
        if item["model"] == "qwen3-4b-instruct-2507"
    )

    assert profile["inherit"] == "qwen.generate"
    assert profile["workload"]["runtime"]["cuda_graphs"] is True
    assert profile["baseline"]["precision"] == "fp16"
