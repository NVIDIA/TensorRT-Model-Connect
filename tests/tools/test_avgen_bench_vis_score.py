# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools import avgen_bench_vis_score as score


def _case(tmp_path: Path, sample_id: str, *, valid: bool = True) -> tuple[dict, dict]:
    frame_paths = []
    for index in score.EXPECTED_RETAINED_FRAME_INDICES:
        path = tmp_path / sample_id / f"frame_{index:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", score.EXPECTED_FRAME_SIZE, color=(index, 0, 0)).save(path)
        frame_paths.append(str(path))
    response = {
        "sample_id": sample_id,
        "stage_output": {
            "data": {
                "returncode": 0,
                "frame_paths": frame_paths,
                "receipt": {
                    "status": "passed",
                    "shape": score.EXPECTED_SHAPE,
                    "retained_frame_indices": score.EXPECTED_RETAINED_FRAME_INDICES,
                },
            }
        },
    }
    if not valid:
        response["stage_output"]["data"]["receipt"]["shape"] = [1, 1, 1, 3]
    request = {
        "sample_id": sample_id,
        "source_category": "ads",
        "source_index": 0,
    }
    return response, request


def test_score_avgen_vis_predictions_applies_only_aggregate_quality_gate(
    tmp_path: Path,
) -> None:
    first_response, first_request = _case(tmp_path, "ads-000")
    second_response, second_request = _case(tmp_path, "ads-001")
    values = iter((0.7, 0.9))

    summary = score.score_avgen_vis_predictions(
        {"responses": [first_response, second_response]},
        {"requests": [first_request, second_request]},
        scorer=lambda _frames: next(values),
        gates={
            "required_sample_count": 2,
            "min_structural_pass_rate": 1.0,
            "min_avgen_vis_mean": 0.8,
        },
    )

    assert summary["status"] == "passed"
    assert summary["valid_count"] == 2
    assert summary["structural_pass_rate"] == 1.0
    assert summary["avgen_vis_mean"] == 0.8
    assert [sample["avgen_vis"] for sample in summary["samples"]] == [0.7, 0.9]


def test_score_avgen_vis_predictions_fails_closed_on_structural_error(
    tmp_path: Path,
) -> None:
    response, request = _case(tmp_path, "ads-000", valid=False)

    summary = score.score_avgen_vis_predictions(
        {"responses": [response]},
        {"requests": [request]},
        scorer=lambda _frames: 1.0,
        gates={
            "required_sample_count": 1,
            "min_structural_pass_rate": 1.0,
            "min_avgen_vis_mean": 0.8,
        },
    )

    assert summary["status"] == "failed"
    assert summary["valid_count"] == 0
    assert {failure["gate"] for failure in summary["gate_failures"]} == {
        "min_structural_pass_rate",
        "min_avgen_vis_mean",
    }
    assert "candidate shape" in summary["samples"][0]["error"]


def test_validate_evaluator_checkout_accepts_pinned_avgen_fixture(
    tmp_path: Path, monkeypatch
) -> None:
    qalign_root = tmp_path / "eval" / "Q-Align"
    scorer_path = qalign_root / "q_align" / "evaluate" / "scorer.py"
    scorer_path.parent.mkdir(parents=True)
    scorer_path.write_text("scorer fixture\n", encoding="utf-8")
    license_path = qalign_root / "S-Lab-LICENSE"
    license_path.write_text("license fixture\n", encoding="utf-8")
    monkeypatch.setattr(score, "QALIGN_SCORER_SHA256", score._sha256(scorer_path))
    monkeypatch.setattr(score, "QALIGN_LICENSE_SHA256", score._sha256(license_path))
    values = {
        "HEAD": score.AVGEN_REVISION,
        f"{score.AVGEN_REVISION}:eval/Q-Align": score.QALIGN_TREE,
    }
    monkeypatch.setattr(score, "_git_value", lambda _root, revision: values[revision])

    assert score.validate_evaluator_checkout(tmp_path) == qalign_root


def test_cli_summary_is_json_serializable(tmp_path: Path) -> None:
    response, request = _case(tmp_path, "ads-000")
    summary = score.score_avgen_vis_predictions(
        {"responses": [response]},
        {"requests": [request]},
        scorer=lambda _frames: 0.85,
        gates={"required_sample_count": 1},
    )

    assert json.loads(json.dumps(summary))["avgen_vis_mean"] == 0.85
