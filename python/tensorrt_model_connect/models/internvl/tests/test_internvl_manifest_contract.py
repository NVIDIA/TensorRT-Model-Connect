# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternVL-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest


@pytest.mark.parametrize("manifest_name", ["internvl3-2b.json", "internvl3-2b-tp2.json"])
def test_internvl3_2b_manifest_declares_hf_image_text_to_text_contract(
    manifest_name: str,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    case = load_manifest(manifest_path)

    assert case.hf_id == "OpenGVLab/InternVL3-2B-hf"
    assert case.task_strategy == "vision_language_generation"
    assert case.user_contract == "image-text-to-text"


@pytest.mark.parametrize(
    "manifest_name",
    ("internvl3-8b.json", "internvl3-8b-tp4.json"),
)
def test_internvl3_8b_builds_reserve_an_exclusive_gpu(
    manifest_name: str,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
