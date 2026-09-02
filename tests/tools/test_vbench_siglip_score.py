# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools import vbench_siglip_score as score


class _TensorLike:
    def float(self):
        return self


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
    prompt_path = tmp_path / sample_id / "prompt.json"
    prompt_path.write_text(json.dumps({"prompt": "a red car moves"}), encoding="utf-8")
    request = {
        "sample_id": sample_id,
        "prompt": "a red car moves",
        "selection_dimension": "motion_smoothness",
        "source_index": 257,
        "inputs": {
            "prompt_file": str(prompt_path),
            "validation_mode": "vbench_siglip",
        },
    }
    return response, request


def test_score_vbench_siglip_reports_metrics_without_uncalibrated_quality_gate(
    tmp_path: Path,
) -> None:
    first_response, first_request = _case(tmp_path, "vbench-000")
    second_response, second_request = _case(tmp_path, "vbench-001")
    values = iter(
        (
            {
                "siglip_alignment": 0.2,
                "temporal_consistency": 0.8,
                "motion_l1": 0.1,
            },
            {
                "siglip_alignment": 0.4,
                "temporal_consistency": 0.9,
                "motion_l1": 0.2,
            },
        )
    )

    summary = score.score_vbench_siglip_predictions(
        {"responses": [first_response, second_response]},
        {"requests": [first_request, second_request]},
        scorer=lambda prompt, _frames: next(values) if prompt else {},
        gates={"required_sample_count": 2, "min_structural_pass_rate": 1.0},
    )

    assert summary["status"] == "passed"
    assert summary["valid_count"] == 2
    assert summary["structural_pass_rate"] == 1.0
    assert summary["metrics"] == {
        "siglip_alignment": {"mean": 0.30000000000000004, "min": 0.2, "max": 0.4},
        "temporal_consistency": {"mean": 0.8500000000000001, "min": 0.8, "max": 0.9},
        "motion_l1": {"mean": 0.15000000000000002, "min": 0.1, "max": 0.2},
    }
    assert summary["calibration_status"] == "pending_reference_baseline"
    assert summary["quality_gate_status"] == "report_only"
    assert summary["gates"] == {
        "required_sample_count": 2,
        "min_structural_pass_rate": 1.0,
    }


def test_score_vbench_siglip_applies_quality_gates_when_explicitly_calibrated(
    tmp_path: Path,
) -> None:
    response, request = _case(tmp_path, "vbench-000")

    summary = score.score_vbench_siglip_predictions(
        {"responses": [response]},
        {"requests": [request]},
        scorer=lambda _prompt, _frames: {
            "siglip_alignment": 0.2,
            "temporal_consistency": 0.8,
            "motion_l1": 0.1,
        },
        gates={
            "required_sample_count": 1,
            "min_structural_pass_rate": 1.0,
            "min_siglip_alignment_mean": 0.3,
        },
    )

    assert summary["status"] == "failed"
    assert summary["calibration_status"] == "quality_gated"
    assert summary["quality_gate_status"] == "configured"
    assert summary["gate_failures"] == [
        {
            "gate": "min_siglip_alignment_mean",
            "actual": 0.2,
            "required": 0.3,
        }
    ]


def test_score_vbench_siglip_fails_closed_on_structural_error(tmp_path: Path) -> None:
    response, request = _case(tmp_path, "vbench-000", valid=False)

    summary = score.score_vbench_siglip_predictions(
        {"responses": [response]},
        {"requests": [request]},
        scorer=lambda _prompt, _frames: {
            "siglip_alignment": 1.0,
            "temporal_consistency": 1.0,
            "motion_l1": 0.1,
        },
        gates={"required_sample_count": 1, "min_structural_pass_rate": 1.0},
    )

    assert summary["status"] == "failed"
    assert summary["valid_count"] == 0
    assert {failure["gate"] for failure in summary["gate_failures"]} == {"min_structural_pass_rate"}
    assert "candidate shape" in summary["samples"][0]["error"]


def test_validate_model_snapshot_accepts_pinned_fixture(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshots" / score.SIGLIP_REVISION
    snapshot.mkdir(parents=True)
    hashes = {}
    for name in ("README.md", "config.json", "model.safetensors"):
        path = snapshot / name
        path.write_text(f"{name} fixture\n", encoding="utf-8")
        hashes[name] = score._sha256(path)
    monkeypatch.setattr(score, "SIGLIP_FILE_SHA256", hashes)

    assert score.validate_model_snapshot(snapshot) == snapshot.resolve()


def test_pooled_feature_tensor_accepts_transformers_5_model_output() -> None:
    tensor = _TensorLike()

    assert score._pooled_feature_tensor(tensor) is tensor
    assert score._pooled_feature_tensor(SimpleNamespace(pooler_output=tensor)) is tensor


def test_cli_summary_is_json_serializable(tmp_path: Path) -> None:
    response, request = _case(tmp_path, "vbench-000")
    summary = score.score_vbench_siglip_predictions(
        {"responses": [response]},
        {"requests": [request]},
        scorer=lambda _prompt, _frames: {
            "siglip_alignment": 0.25,
            "temporal_consistency": 0.9,
            "motion_l1": 0.05,
        },
        gates={"required_sample_count": 1},
    )

    encoded = json.loads(json.dumps(summary))
    assert encoded["metrics"]["siglip_alignment"]["mean"] == 0.25
