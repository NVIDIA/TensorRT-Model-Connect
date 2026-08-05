# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import pytest

from tools import full_duplex_bench_score as fdb_score
from tools import prepare_full_duplex_bench_validation as prepare_fdb


def _score(*, tor_shift: float = 0.0, frequency_shift: float = 0.0, jsd_shift: float = 0.0):
    metrics = {}
    cases = []
    for category, fields in fdb_score.METRIC_FIELDS.items():
        case = {"category": category, "sample_id": f"{category}-0"}
        for field in fields:
            value = {
                "tor": 0.2 + tor_shift,
                "frequency": 0.03 + frequency_shift,
                "jsd": 0.8 + jsd_shift,
            }[field]
            metrics[f"{category}.{field}"] = {
                "value": value,
                "stddev": 0.0,
                "sample_count": 30,
            }
            case[field] = value
        cases.append(case)
    return {"metrics": metrics, "cases": cases}


GATES = {
    "max_tor_abs_delta": 0.10,
    "max_backchannel_frequency_abs_delta": 0.01,
    "max_backchannel_jsd_abs_delta": 0.02,
}


def _answers(*, samples_per_category: int = 30):
    return {
        "schema_version": "trtmc.full-duplex-bench-validation/v1",
        "source_revision": prepare_fdb.FDB_REVISION,
        "sampling": {"seed": prepare_fdb.SELECTION_SEED},
        "requests": [
            {"sample_id": f"{category}-{index}", "category": category}
            for category in fdb_score.METRIC_FIELDS
            for index in range(samples_per_category)
        ],
    }


def test_formal_validation_requires_fixed_30_per_category_slice() -> None:
    assert fdb_score.FDB_REVISION == prepare_fdb.FDB_REVISION
    assert fdb_score.SELECTION_SEED == prepare_fdb.SELECTION_SEED
    requests = fdb_score.validate_requests_manifest(_answers())

    assert len(requests) == 150


def test_formal_validation_rejects_a_smaller_debug_slice() -> None:
    with pytest.raises(ValueError, match="exactly 30 samples per category"):
        fdb_score.validate_requests_manifest(_answers(samples_per_category=20))


def test_request_input_requires_the_prepared_audio_checksum(tmp_path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fixed prepared audio")
    request = {
        "inputs": {"audio": str(audio)},
        "prepared_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
    }

    assert fdb_score._request_input(request) == audio

    audio.write_bytes(b"changed audio")
    with pytest.raises(ValueError, match="checksum does not match"):
        fdb_score._request_input(request)


def test_tor_uses_the_benchmark_word_count_and_duration_boundaries() -> None:
    assert fdb_score._tor([]) == 0
    assert fdb_score._tor(
        [
            {"text": "yes", "timestamp": [0.0, 0.2]},
            {"text": "right", "timestamp": [0.3, 0.8]},
        ]
    ) == 0
    assert fdb_score._tor(
        [
            {"text": "this", "timestamp": [0.0, 0.2]},
            {"text": "is", "timestamp": [0.2, 0.4]},
            {"text": "too", "timestamp": [0.4, 0.6]},
            {"text": "long", "timestamp": [0.6, 0.8]},
        ]
    ) == 1


def test_overlap_words_includes_each_supported_overlap_shape() -> None:
    chunks = [
        {"text": "inside", "timestamp": [1.1, 1.2]},
        {"text": "left", "timestamp": [0.8, 1.1]},
        {"text": "right", "timestamp": [1.9, 2.1]},
        {"text": "outside", "timestamp": [2.1, 2.2]},
    ]

    assert fdb_score._overlap_words(chunks, 1.0, 2.0) == [
        "inside",
        "left",
        "right",
    ]


def test_backchannel_metrics_cover_silence_and_long_speech() -> None:
    ground_truth = [0.5, 0.5]

    silence = fdb_score._backchannel_metrics(
        segments=[], chunks=[], duration=2.0, ground_truth=ground_truth
    )
    assert silence == {"tor": 0.0, "frequency": 0.0, "jsd": 1.0}

    long_speech = fdb_score._backchannel_metrics(
        segments=[{"start": 0.0, "end": 3.1}],
        chunks=[],
        duration=4.0,
        ground_truth=ground_truth,
    )
    assert long_speech == {"tor": 1.0, "frequency": 0.0, "jsd": 1.0}


def test_backchannel_metrics_match_pinned_definition_golden() -> None:
    result = fdb_score._backchannel_metrics(
        segments=[{"start": 0.0, "end": 0.8}],
        chunks=[
            {"text": "uh", "timestamp": [0.1, 0.2]},
            {"text": "huh", "timestamp": [0.3, 0.5]},
        ],
        duration=2.0,
        ground_truth=[0.2, 0.6, 0.2],
    )

    assert result["tor"] == 0.0
    assert result["frequency"] == pytest.approx(0.5)
    # Golden value from the pinned Full-Duplex-Bench 0.2-second histogram,
    # linear-resize, and Jensen-Shannon-distance definitions.
    assert result["jsd"] == pytest.approx(0.5143306819821533)


def test_compare_scores_passes_each_backend_metric_within_budget() -> None:
    result = fdb_score.compare_scores(
        _score(),
        _score(tor_shift=0.05, frequency_shift=-0.005, jsd_shift=0.01),
        gates=GATES,
    )

    assert result["status"] == "passed"
    assert result["metric_gate_count"] == 7
    assert result["passed_count"] == 7
    assert result["metric_gate_pass_rate"] == 1.0
    metric = result["metrics"]["icc_backchannel.frequency"]
    assert metric["hf"] == pytest.approx(0.03)
    assert metric["trtmc"] == pytest.approx(0.025)
    assert metric["trtmc_minus_hf"] == pytest.approx(-0.005)
    assert metric["abs_delta"] == pytest.approx(0.005)
    assert metric["threshold"] == pytest.approx(0.01)
    assert metric["passed"] is True
    assert metric["sample_count"] == 30
    assert metric["paired_changed_count"] == 1
    assert metric["paired_mean_abs_delta"] == pytest.approx(0.005)
    assert metric["paired_max_abs_delta"] == pytest.approx(0.005)


def test_compare_scores_reports_metric_and_representative_failure() -> None:
    result = fdb_score.compare_scores(
        _score(),
        _score(jsd_shift=0.03),
        gates=GATES,
    )

    assert result["status"] == "failed"
    assert result["passed_count"] == 6
    assert len(result["gate_failures"]) == 1
    failure = result["gate_failures"][0]
    assert failure["metric"] == "icc_backchannel.jsd"
    assert failure["value"] == pytest.approx(0.03)
    assert failure["threshold"] == pytest.approx(0.02)
    assert failure["operator"] == "<="
    assert result["disagreements"][0]["sample_id"] == "icc_backchannel-0"
