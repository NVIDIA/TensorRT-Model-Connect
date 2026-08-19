# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_phi3_mini_uses_accuracy_preserving_decoder_layout() -> None:
    manifest = Path(__file__).with_name("manifests") / "phi3-mini.json"
    case = load_manifest(manifest)

    assert case.metadata["build_args"]["decoder_engine_layout"] == "dual_profile"


def test_phi3_mini_reference_matches_bundle_precision() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "phi3-mini.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp16"
    assert manifest["testcases"][0]["reference_precision"] == manifest["precision"]
