# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure the overflow regression inputs exercise the reported dynamic range."""

import numpy as np

from families.timesfm.tests.overflow import histories
from families.timesfm.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_histories_exceed_fp16_only_after_first_patch_normalization() -> None:
    for case in histories(2048, 32):
        values = np.asarray(case["past_values"], dtype=np.float32)
        assert np.isfinite(values).all() and np.max(np.abs(values)) < np.finfo(np.float16).max
        first = values[:32]
        normalized = (values - first.mean()) / max(first.std(), 1e-6)
        assert np.max(np.abs(normalized)) > 1e8


def test_default_precision_remains_fp32() -> None:
    support = describe(ModelMetadata({"model_type": "timesfm"}, {}))
    assert support is not None and support.default_precision == "fp32"
