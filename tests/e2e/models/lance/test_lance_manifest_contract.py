# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_lance_x2t_manifest_declares_hf_any_to_any_contract() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "lance-3b-x2t-image.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "bytedance-research/Lance"
    assert case.task_strategy == "vision_language_generation"
    assert case.user_contract == "any-to-any"
    assert "skip_reason" not in case.metadata
    assert case.reference_backend == "golden_snapshot"
    expected_snapshot = manifest_path.parent.parent / "data/lance_3b_x2t_golden.json"
    assert Path(case.metadata["golden_snapshot_path"]) == expected_snapshot
