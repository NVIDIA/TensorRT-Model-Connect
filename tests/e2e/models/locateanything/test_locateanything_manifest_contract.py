# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path
import json

from tests.e2e_harness.manifest_loader import load_manifest


def test_locateanything_manifest_declares_hf_image_text_to_text_contract() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "locateanything-3b.json"
    case = load_manifest(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert case.hf_id == "nvidia/LocateAnything-3B"
    assert case.task_strategy == "vision_language_generation"
    assert case.user_contract == "image-text-to-text"
    assert case.inputs["prompt"] == "Find the white vehicle in this image."
    assert "precision" not in raw
    assert "max_cache_length" not in raw
