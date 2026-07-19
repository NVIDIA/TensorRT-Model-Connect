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
