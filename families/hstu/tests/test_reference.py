# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The independent oracle must preserve small terms in cancelling dense sums."""

import numpy as np
from ..config import Config
from ..reference import Reference


def test_dense_reference_preserves_cancelling_sum():
    oracle = Reference(
        Config(),
        {
            "projection.weight": np.ones((1, 3), np.float32),
            "projection.bias": np.zeros(1, np.float32),
        },
    )
    x = np.asarray([[1e8, 1, -1e8]], np.float32)
    np.testing.assert_array_equal(oracle.linear(x, "projection."), [[1]])
