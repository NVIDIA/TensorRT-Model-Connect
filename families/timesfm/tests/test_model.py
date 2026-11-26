# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder rejects retired or unimplemented tasks before reading weights."""

from pathlib import Path

import pytest

pytest.importorskip("tensorrt")

from families.timesfm import model
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize("task", ["time_series_forecast", "series_to_quantile_forecast",
                                 "series_to_point_and_quantile_forecast"])
def test_build_rejects_other_tasks(tmp_path: Path, task: str):
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "unused.bundle",
                           family="timesfm", task=task, precision="fp32")
    with pytest.raises(ValueError, match="supports only series_to_point_forecast"):
        model.build(request, object())
