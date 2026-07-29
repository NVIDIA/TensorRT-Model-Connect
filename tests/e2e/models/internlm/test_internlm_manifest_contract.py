# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM-owned native KV manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_official_manifest_uses_family_build_defaults() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests") / "internlm2-1.8b.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert manifest["hf_id"] == "internlm/internlm2-math-plus-1_8b"
    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert case.metadata["reference_precision"] == "fp32"


def test_all_internlm_manifests_removed_legacy_build_flags() -> None:
    manifest_dir = Path(__file__).with_name("manifests")
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "precision" not in manifest, manifest_path.name
        assert "max_cache_length" not in manifest, manifest_path.name
