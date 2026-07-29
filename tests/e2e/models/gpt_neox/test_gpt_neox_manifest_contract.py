# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-NeoX-owned native KV manifest contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_pythia_manifest_uses_family_build_defaults() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests") / "pythia-70m.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert manifest["hf_id"] == "EleutherAI/pythia-70m"
    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert manifest["reference_precision"] == "fp32"

    testcase = manifest["testcases"][0]
    assert hashlib.sha256(testcase["prompt"].encode()).hexdigest() == (
        "61f41777dda57d6f63957816782b22ea5951a914e04c60ee15ca2235bdb1eb0e"
    )
    assert testcase["max_new_tokens"] == 32


def test_gpt_neox_manifests_removed_legacy_build_flags() -> None:
    manifest_dir = Path(__file__).with_name("manifests")
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "precision" not in manifest, manifest_path.name
        assert "max_cache_length" not in manifest, manifest_path.name
        assert "build_args" not in manifest, manifest_path.name
