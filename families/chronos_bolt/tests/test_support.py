# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for chronos_bolt."""

from families.chronos_bolt.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(
        ModelMetadata(
            config={
                "model_type": "chronos_bolt",
                "architectures": ["ChronosBoltModelForForecasting"],
            },
            model_index={},
        )
    )
    assert support is not None
    assert support.tasks == ("series_to_quantile_forecast",)
    assert support.default_task == "series_to_quantile_forecast"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "unrelated-model"}, model_index={})) is None
