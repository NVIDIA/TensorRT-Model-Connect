# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-visible TimesFM checkpoint identity and semantic Task contracts."""

import pytest

from families.timesfm.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("model_type", ["timesfm", "times_fm"])
def test_checkpoint_resolves_to_point_forecast(model_type):
    metadata = ModelMetadata({"model_type": model_type}, {})
    family, support = resolve_family(metadata, requested_family="timesfm")
    assert family == "timesfm"
    assert support.tasks == ("series_to_point_forecast",)
    assert support.default_task == "series_to_point_forecast"


def test_unrelated_checkpoint_is_not_claimed():
    assert describe(ModelMetadata({"model_type": "not_timesfm"}, {})) is None
