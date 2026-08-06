# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path


def test_nemotron_hindi_reference_matches_bundle_precision() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "nemotron-hindi-4b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert manifest["testcases"][0]["reference_precision"] == manifest["precision"]
