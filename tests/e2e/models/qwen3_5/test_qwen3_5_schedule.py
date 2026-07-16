# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-owned scheduler classification checks."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import find_manifest_path
from tools.ci import e2e_schedule as schedule_e2e


_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_qwen35_is_marked_exclusive_gpu() -> None:
    models_dir = _REPO_ROOT / "tests" / "e2e" / "models"
    manifest_path = find_manifest_path("qwen35-9b", models_dir)
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text())

    assert schedule_e2e.classify_parallel_resource(manifest) == "exclusive_gpu"
