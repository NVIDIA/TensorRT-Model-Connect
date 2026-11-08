# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint-specific PatchTST Task discovery."""

import pytest

from families.patchtst.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize(
    "config,task",
    [
        ({}, "series_to_point_forecast"),
        ({"architectures": ["PatchTSTForPrediction"], "loss": "mse"}, "series_to_point_forecast"),
        (
            {"architectures": ["PatchTSTForRegression"], "loss": "mse"},
            "series_to_regression_values",
        ),
        *[
            (
                {
                    "architectures": ["PatchTSTForRegression"],
                    "loss": loss,
                    "distribution_output": distribution,
                },
                "series_to_regression_distribution",
            )
            for loss in (None, "nll")
            for distribution in ("normal", "student_t", "negative_binomial")
        ],
    ],
)
def test_loaded_checkpoint_has_exactly_its_own_task(config, task):
    metadata = ModelMetadata({"model_type": "patchtst", **config}, {})
    family, support = resolve_family(metadata)
    assert family == "patchtst"
    assert support.tasks == (task,)
    assert support.default_task == task
    assert support.default_precision == "fp32"


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "not_patchtst"},
        {"model_type": "patchtst", "architectures": ["PatchTSTForClassification"]},
        {"model_type": "patchtst", "task_type": "classification"},
        {
            "model_type": "patchtst",
            "architectures": ["PatchTSTForRegression"],
            "distribution_output": "unsupported",
        },
    ],
)
def test_unimplemented_checkpoint_modes_are_not_advertised(config):
    assert describe(ModelMetadata(config, {})) is None
