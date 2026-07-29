# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "manifest_name",
    ("deepseek-v2-lite.json", "deepseek-v2-lite-tp4.json"),
)
def test_deepseek_v2_lite_builds_reserve_an_exclusive_gpu(
    manifest_name: str,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_deepseek_v2_lite_has_a_precision_matched_l0() -> None:
    model_dir = Path(__file__).parent
    lite = json.loads(
        (model_dir / "manifests" / "deepseek-v2-lite.json").read_text(encoding="utf-8")
    )
    replacement = json.loads(
        (
            model_dir
            / "manifests"
            / "deepseek-v2-tiny-fp16-l0.json"
        ).read_text(encoding="utf-8")
    )
    replacement_threshold = json.loads(
        (
            model_dir
            / "thresholds"
            / "deepseek-v2-tiny-fp16-l0.json"
        ).read_text(encoding="utf-8")
    )
    lite_threshold = json.loads(
        (model_dir / "thresholds" / "deepseek-v2-lite.json").read_text(encoding="utf-8")
    )

    assert lite["testcases"][0]["l0_replacement"] == replacement["name"]
    for field in ("family", "runtime_strategy", "precision", "quantization"):
        assert lite.get(field) == replacement.get(field)
    assert replacement["hf_id"] == "katuni4ka/tiny-random-deepseek-v3"
    assert replacement["hf_revision"] == "ba144b0d3331a5892aa588d82722d382be2b6e6b"
    assert replacement["bundle"] == "deepseek-v2-tiny-fp16-l0.trtfb"
    assert replacement["e2e_parallel_resource"] == "exclusive_gpu"
    assert replacement["testcases"][0]["ci_tier"] == "l0_only"
    assert replacement["testcases"][0]["reference_precision"] == "fp32"
    for field in ("prompt", "max_new_tokens"):
        assert replacement["testcases"][0].get(field) == lite["testcases"][0].get(field)
    assert replacement_threshold == lite_threshold
