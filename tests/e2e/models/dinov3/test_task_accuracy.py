# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tests.e2e.models.dinov3.e2e_plugins.comparator import comparator
from tests.e2e.models.dinov3.e2e_plugins.knn import tensor_payload
from tests.e2e_harness.contracts import (
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _output(
    features: np.ndarray, labels: list[int], predictions: dict[str, list[int]]
) -> StageOutput:
    return StageOutput(
        stage_name="full_inference",
        data={
            "knn_task_accuracy": True,
            "bank_size": 1034,
            "query_count": len(labels),
            "class_names": ["angular_leaf_spot", "bean_rust", "healthy"],
            "labels": labels,
            "predictions": predictions,
            "query_pooler_output": tensor_payload(features),
        },
    )


def _predictions(values: list[int]) -> dict[str, list[int]]:
    return {str(k): list(values) for k in (10, 20, 100, 200)}


def _metric_case(sample_id: str, result) -> dict:
    return {
        "sample_id": sample_id,
        "metrics": {
            name: {"value": metric.value} for name, metric in result.metrics.items()
        },
    }


def _gates() -> dict:
    return {
        "exact_task_query_count": 4,
        "min_reference_20nn_top1_accuracy": 0.5,
        "max_candidate_20nn_top1_accuracy_drop_from_reference": 0.01,
        "min_candidate_reference_20nn_top1_agreement": 0.98,
        "min_task_query_pooler_cosine": 0.999,
        "max_task_query_pooler_relative_l2": 0.01,
    }


def test_knn_comparison_records_task_sufficient_statistics() -> None:
    labels = [0, 1, 2, 0]
    reference_predictions = _predictions([0, 1, 2, 1])
    candidate_predictions = _predictions([0, 1, 2, 1])
    features = np.eye(4, dtype=np.float32)

    result = comparator.compare(
        _output(features, labels, candidate_predictions),
        _output(features, labels, reference_predictions),
        ThresholdProfile(task_strategy="image_feature_extraction", metrics={}),
        StageSpec(name="full_inference", required=True),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["query_count"].value == 4
    assert result.metrics["candidate_correct_20nn"].value == 3
    assert result.metrics["reference_correct_20nn"].value == 3
    assert result.metrics["prediction_agreement_20nn"].value == 4


def test_knn_aggregate_gates_complete_ground_truth_task_accuracy() -> None:
    labels = [0, 1, 2, 0]
    features = np.eye(4, dtype=np.float32)
    result = comparator.compare(
        _output(features, labels, _predictions([0, 1, 2, 1])),
        _output(features, labels, _predictions([0, 1, 2, 1])),
        ThresholdProfile(task_strategy="image_feature_extraction", metrics={}),
        StageSpec(name="full_inference", required=True),
    )

    aggregate = comparator.aggregate([_metric_case("all", result)], _gates())

    assert aggregate["passed"] is True
    assert aggregate["task_accuracy"]["task_query_count"] == 4
    assert aggregate["task_accuracy"]["candidate_20nn_top1_accuracy"] == 0.75
    assert aggregate["task_accuracy"]["reference_20nn_top1_accuracy"] == 0.75
    assert aggregate["task_accuracy"]["candidate_reference_20nn_top1_agreement"] == 1.0
    assert (
        aggregate["task_accuracy"][
            "candidate_20nn_top1_accuracy_drop_from_reference"
        ]
        == 0.0
    )


def test_knn_aggregate_fails_candidate_accuracy_drop() -> None:
    labels = [0, 1, 2, 0]
    features = np.eye(4, dtype=np.float32)
    result = comparator.compare(
        _output(features, labels, _predictions([1, 1, 2, 1])),
        _output(features, labels, _predictions([0, 1, 2, 1])),
        ThresholdProfile(task_strategy="image_feature_extraction", metrics={}),
        StageSpec(name="full_inference", required=True),
    )

    aggregate = comparator.aggregate([_metric_case("all", result)], _gates())

    assert aggregate["passed"] is False
    assert aggregate["gate_results"]["candidate_20nn_top1_accuracy"] is False
    assert aggregate["gate_results"]["candidate_reference_20nn_top1_agreement"] is False
