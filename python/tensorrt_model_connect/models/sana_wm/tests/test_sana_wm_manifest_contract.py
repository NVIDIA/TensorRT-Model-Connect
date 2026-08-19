# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.manifest_loader import load_model_manifest


def test_sana_wm_manifest_preserves_exact_model_card_contract() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "sana-wm-bidirectional.json"
    model = load_model_manifest(manifest_path)

    assert model.name == "sana-wm-bidirectional"
    assert len(model.testcases) == 1

    case = model.testcases[0]
    assert case.name == model.name
    assert case.inputs["video_num_frames"] == 321
    assert case.inputs["num_inference_steps"] == 60
    assert case.inputs["cfg_scale"] == 5.0
    assert case.inputs["flow_shift"] == 9.8
    assert case.metadata.get("ci_tier", "") == ""
    assert case.threshold_overrides["contract_min_frame_count"] == 320
    assert case.threshold_overrides["contract_max_frame_count_delta"] == 0
    assert case.threshold_overrides["contract_exact_frames"] == 1


def test_sana_wm_build_timeout_covers_the_full_model_card_build() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "sana-wm-bidirectional.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = load_model_manifest(manifest_path)

    assert manifest["build_timeout_s"] == 7200
    assert model.testcases[0].metadata["build_timeout_s"] == 7200
