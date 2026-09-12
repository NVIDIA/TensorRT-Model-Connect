# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DETR family checkpoint weight container."""

from __future__ import annotations

import numpy as np

from families.detr.weights import WeightDict


def test_weight_dict_is_a_concrete_mapping():
    weights = WeightDict()
    weights["model.input_projection.weight"] = np.zeros((2, 3), dtype=np.float32)

    assert isinstance(weights, dict)
    assert weights["model.input_projection.weight"].shape == (2, 3)
