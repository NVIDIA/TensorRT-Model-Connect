# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-owned native KV manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_native_l0_manifest_uses_supported_official_checkpoint_and_family_defaults() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests") / "riva-translate-4b-l0.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert manifest["hf_id"] == "nvidia/Riva-Translate-4B-Instruct-v1.1"
    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert case.metadata["reference_precision"] == "fp32"
    assert case.metadata["ci_tier"] == "l0_only"
    assert case.threshold_overrides["contract_token_agreement_rate"] == 1.0


def test_mistral_7b_nightly_uses_full_attention_checkpoint() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "mistral-7b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert manifest["hf_id"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert case.metadata["ci_tier"] == "nightly_only"
    assert case.metadata["l0_replacement"] == "riva-translate-4b-l0"
    assert case.threshold_overrides["contract_token_agreement_rate"] == 1.0


def test_all_mistral_manifests_removed_legacy_build_flags() -> None:
    manifest_dir = Path(__file__).with_name("manifests")
    for manifest_path in sorted(manifest_dir.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "precision" not in manifest, manifest_path.name
        assert "max_cache_length" not in manifest, manifest_path.name
