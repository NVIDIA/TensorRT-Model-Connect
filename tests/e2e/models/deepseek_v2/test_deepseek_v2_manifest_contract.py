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


def test_deepseek_v2_l0_replacement_preserves_precision() -> None:
    manifests = Path(__file__).with_name("manifests")
    lite = json.loads(
        (manifests / "deepseek-v2-lite.json").read_text(encoding="utf-8")
    )
    tiny = json.loads(
        (manifests / "deepseek-v2-tiny.json").read_text(encoding="utf-8")
    )

    assert lite["testcases"][0]["l0_replacement"] == tiny["name"]
    assert lite["precision"] == tiny["precision"] == "fp16"
    assert tiny["testcases"][0]["reference_precision"] == tiny["precision"]
    assert tiny["task_eval"]["hf_experts_implementation"] == "batched_mm"
