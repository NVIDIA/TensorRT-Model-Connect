# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AIPerf dataset-qualification checker for DINOv2 features."""

import numpy as np

from families.dinov2.validation import compare


def test_dinov2_checker_applies_the_e2e_feature_tolerance_of_the_build_precision():
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(1, 5, 8))

    def side(values):
        return {"last_hidden_state": values.reshape(-1).tolist(), "last_hidden_state_shape": [1, 5, 8],
                "pooler_output": values[:, 0].reshape(-1).tolist(), "pooler_output_shape": [1, 8]}

    assert compare(side(hidden + 1e-3), side(hidden), {"precision": "fp32"})[0] is True
    assert compare(side(hidden + 1e-2), side(hidden), {"precision": "fp32"})[0] is False
    assert compare(side(hidden + 1e-2), side(hidden), {"precision": "fp16"})[0] is True
    assert compare(side(hidden + 1.5e-2), side(hidden), {"precision": "fp32", "threshold_scale": 2.0})[0] is False
    assert compare(None, side(hidden), {"precision": "fp32"})[0] is None
