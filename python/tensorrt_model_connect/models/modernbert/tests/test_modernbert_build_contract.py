# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for ModernBERT."""

from __future__ import annotations

import json
from pathlib import Path


def test_acceptance_build_uses_fp32_for_representation_parity() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "modernbert-base.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert manifest["testcases"][0]["reference_precision"] == "fp32"
