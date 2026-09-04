# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.video_parity_shadow import (
    SCHEMA_VERSION,
    VideoPair,
    _cgvqm_preprocess,
    _flow_consistency_from_fields,
    _open_pair,
    load_pair_manifest,
    stratified_frame_indices,
    summarize,
)


def test_stratified_frame_indices_are_unique_and_endpoint_inclusive() -> None:
    assert stratified_frame_indices(10, 4) == [0, 3, 6, 9]
    assert stratified_frame_indices(3, 9) == [0, 1, 2]
    with pytest.raises(ValueError, match="positive"):
        stratified_frame_indices(10, 0)


def test_metric_summary_reports_tail_instead_of_only_mean() -> None:
    result = summarize([0.0, 0.0, 0.0, 1.0])
    assert result.count == 4
    assert result.mean == pytest.approx(0.25)
    assert result.median == pytest.approx(0.0)
    assert result.p95 == pytest.approx(0.85)
    assert result.maximum == pytest.approx(1.0)


def test_pair_manifest_binds_labels_and_resolves_relative_paths(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npy"
    candidate = tmp_path / "candidate.npy"
    frames = np.zeros((2, 4, 8, 3), dtype=np.uint8)
    np.save(reference, frames)
    np.save(candidate, frames)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "pairs": [
                    {
                        "sample_id": "same",
                        "reference": reference.name,
                        "candidate": candidate.name,
                        "expected": "match",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pairs = load_pair_manifest(manifest)

    assert pairs == [
        VideoPair(
            sample_id="same",
            reference=reference.resolve(),
            candidate=candidate.resolve(),
            expected="match",
        )
    ]
    loaded_reference, loaded_candidate = _open_pair(pairs[0])
    assert loaded_reference.shape == loaded_candidate.shape == frames.shape


def test_pair_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    frames = tmp_path / "frames.npy"
    np.save(frames, np.zeros((2, 4, 8, 3), dtype=np.uint8))
    manifest = tmp_path / "pairs.json"
    row = {"sample_id": "duplicate", "reference": frames.name, "candidate": frames.name}
    manifest.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "pairs": [row, row]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_pair_manifest(manifest)


def test_flow_field_consistency_distinguishes_same_motion_from_freeze() -> None:
    reference_fields = [np.full((4, 8, 2), (2.0, 0.0), dtype=np.float32)] * 3
    same = _flow_consistency_from_fields(reference_fields, reference_fields)
    frozen_fields = [np.zeros((4, 8, 2), dtype=np.float32)] * 3
    frozen = _flow_consistency_from_fields(reference_fields, frozen_fields)

    assert same["normalized_endpoint_error"]["maximum"] == pytest.approx(0.0)
    assert same["candidate_to_reference_motion_ratio"] == pytest.approx(1.0)
    assert frozen["normalized_endpoint_error"]["minimum"] > 0.0
    assert frozen["candidate_to_reference_motion_ratio"] == pytest.approx(0.0)


def test_cgvqm_preprocess_normalizes_and_moves_time_after_channels() -> None:
    torch = pytest.importorskip("torch")
    frames = torch.tensor(
        [
            [[[0.43216]], [[0.394666]], [[0.37645]]],
            [[[0.66019]], [[0.616116]], [[0.593439]]],
        ],
        dtype=torch.float32,
    )

    result = _cgvqm_preprocess(frames)

    assert result.shape == (3, 2, 1, 1)
    assert torch.allclose(result[:, 0], torch.zeros((3, 1, 1)), atol=1e-6)
    assert torch.allclose(result[:, 1], torch.ones((3, 1, 1)), atol=1e-6)
