# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
from families.internvl.model import _InternVLModel


def test_bundle_normalization_matches_internvl_imagenet_processor():
    config = SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=16)
    runtime = _InternVLModel().get_vl_config(config)
    black_white = np.asarray([[0.0] * 3, [1.0] * 3])
    actual = (black_white - runtime["image_mean"]) / runtime["image_std"]
    expected = [[-2.11790393, -2.03571429, -1.80444444],
                [2.24890830, 2.42857143, 2.64000000]]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
