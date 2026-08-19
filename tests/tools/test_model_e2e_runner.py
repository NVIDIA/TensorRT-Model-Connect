# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for uniform model-manifest E2E execution."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness import model_runner
from tests.e2e_harness.contracts import E2ECase


MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tensorrt_model_connect"
    / "models"
)


class _Config:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name: str, default=None):
        return self._options.get(name, default)


def _case_matches(_case, _filters) -> bool:
    return True


def _is_multi_device(case) -> bool:
    return case.metadata.get("ci_tier") == "multi_device"


def test_platform_threshold_overrides_are_scoped_to_matching_platform() -> None:
    case = E2ECase(
        name="synthetic-case",
        hf_id="example-org/synthetic-model",
        family="synthetic_family",
        runtime_strategy="synthetic_runtime",
        task_strategy="synthetic_task",
        threshold_overrides={
            "max_error": 0.01,
            "min_score": 0.99,
        },
        metadata={
            "platform_threshold_overrides": {"test-platform": {"max_error": 0.011}}
        },
    )

    platform_case = model_runner._case_with_platform_thresholds(case, "test-platform")
    default_case = model_runner._case_with_platform_thresholds(case, "")

    assert platform_case.threshold_overrides == {
        "max_error": 0.011,
        "min_score": 0.99,
    }
    assert default_case is case
    assert case.threshold_overrides["max_error"] == 0.01


def test_model_collection_applies_worker_partition() -> None:
    options = {
        "--e2e-exclude-ci-tier": [],
        "--e2e-partition-id": 1,
        "--e2e-partition-size": 2,
    }
    names = model_runner.model_names_for_dir(
        config=_Config(**options),
        model_dir=MODELS_DIR,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )
    models = model_runner.load_all_model_manifests(MODELS_DIR)
    selected_models = [
        model
        for model in models
        if model_runner.selected_testcases(
            model,
            config=_Config(**options),
            case_matches_model=_case_matches,
            is_multi_device_case=_is_multi_device,
        )
    ]

    assert (
        names == [model.name for model in sorted(selected_models, key=lambda item: item.name)][1::2]
    )
