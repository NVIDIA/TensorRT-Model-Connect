# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import _append_declared_build_cli_args


def test_nemotron_hindi_reference_matches_bundle_precision() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "nemotron-hindi-4b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "bf16"
    assert manifest["testcases"][0]["reference_precision"] == manifest["precision"]


def test_nemotron_mini_uses_qualified_compact_build() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "nemotron-mini-4b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)
    command: list[str] = []

    assert manifest["build_args"] == {"decoder_engine_layout": "dual_profile"}
    assert manifest["build_cli_args"] == [
        {
            "flag": "--set",
            "value": "nemotron_decoder.builder_workspace_gib=2",
        }
    ]
    _append_declared_build_cli_args(command, case)
    assert command == ["--set", "nemotron_decoder.builder_workspace_gib=2"]
