# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for Llama models."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_falcon3_split_decoder_build_reserves_an_exclusive_gpu() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "falcon3-1b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_native_minitron_manifests_use_family_build_defaults() -> None:
    for manifest_name in (
        "minitron-4b-width.json",
        "minitron-4b-width-l0.json",
    ):
        manifest_path = Path(__file__).parent / "manifests" / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = load_manifest(manifest_path)

        assert "precision" not in manifest, manifest_name
        assert "max_cache_length" not in manifest, manifest_name
        assert "precision" not in case.metadata, manifest_name
        assert "max_cache_length" not in case.inputs, manifest_name


def test_tinyllama_keeps_legacy_build_contract() -> None:
    manifest_path = (
        Path(__file__).parent / "manifests" / "tinyllama-1.1b.json"
    )
    case = load_manifest(manifest_path)

    assert case.metadata["precision"] == "fp16"
    assert case.inputs["max_cache_length"] == 256
