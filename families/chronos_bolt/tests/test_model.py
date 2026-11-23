# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from families.chronos_bolt import model
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def _request(tmp_path: Path, task: str) -> BuildRequest:
    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="chronos_bolt",
        task=task,
        precision="fp32",
    )


def test_checkpoint_resolves_to_the_quantile_forecast_task() -> None:
    family, support = resolve_family(
        ModelMetadata(
            {"model_type": "chronos_bolt", "architectures": ["ChronosBoltModelForForecasting"]},
            {},
        )
    )

    assert family == "chronos_bolt"
    assert support.tasks == ("series_to_quantile_forecast",)
    assert support.default_task == "series_to_quantile_forecast"


@pytest.mark.parametrize(
    "task",
    ["time_series_forecast", "series_to_point_forecast", "series_to_point_and_quantile_forecast"],
)
def test_build_rejects_every_task_but_the_quantile_forecast(tmp_path: Path, task: str) -> None:
    with pytest.raises(ValueError, match="supports only series_to_quantile_forecast"):
        model.build(_request(tmp_path, task), object())
