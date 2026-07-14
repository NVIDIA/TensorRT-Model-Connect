# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for TimesFM."""

from __future__ import annotations

import json
from pathlib import Path


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "timesfm-2.0-500m-official.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_acceptance_build_keeps_late_decoder_tail_in_fp32() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "timesfm-2.0-500m-official.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fp32_layers = set(manifest["fp32_layers"])

    assert set(range(35, 50)) <= fp32_layers
    assert {50, 51, 52, 53} <= fp32_layers
