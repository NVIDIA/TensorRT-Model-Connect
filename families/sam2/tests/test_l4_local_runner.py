# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-free contracts for the explicit SAM2 L4 local runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.sam2.tests import l4_local_runner as local
from families.sam2.tests import test_e2e


def test_local_contract_is_outside_automatic_manifest_inventory() -> None:
    manifest, thresholds = local._contract()

    assert manifest["qualification"] == "local_only"
    assert manifest["name"] == "sam2-l4-local"
    assert "sam2-l4-local" not in test_e2e.CASES
    assert thresholds["minimum_frame_mask_iou"] == 0.98


def test_explicit_paths_fail_closed_without_caller_assets(tmp_path: Path) -> None:
    with pytest.raises(local.QualificationError, match="probe is missing"):
        local.run(
            tmp_path / "probe",
            tmp_path / "sam2-l4-local.bundle",
            tmp_path / "runtime",
            tmp_path / "fixtures",
        )


def test_frame_inventory_rejects_missing_or_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "rgb8"
    root.mkdir()
    for index in range(4):
        (root / f"{index:06d}.rgb8").write_bytes(b"")

    with pytest.raises(local.QualificationError, match="exactly five frames"):
        local._frames(tmp_path)


def test_threshold_gate_rejects_any_accuracy_regression() -> None:
    _, thresholds = local._contract()
    passing = {
        "minimum_frame_mask_iou": 0.98,
        "minimum_macro_mask_iou": 0.99,
        "minimum_global_mask_iou": 0.99,
        "minimum_bbox_iou": 0.995,
        "maximum_bbox_coordinate_error": 0.5,
        "maximum_bbox_score_error": 0.01,
        "label_exact": 1.0,
    }
    local._enforce(passing, thresholds)
    failing = dict(passing)
    failing["minimum_frame_mask_iou"] = 0.979

    with pytest.raises(local.QualificationError, match="minimum_frame_mask_iou"):
        local._enforce(failing, thresholds)


def test_golden_bbox_shape_is_required(tmp_path: Path) -> None:
    root = tmp_path / "golden"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"unexpected": {}}), encoding="utf-8")

    with pytest.raises(local.QualificationError, match="bbox contract"):
        local._golden(tmp_path)
