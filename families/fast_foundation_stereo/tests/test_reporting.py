# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual error scales must not hide a systematic disparity shift."""

import numpy as np

from families.fast_foundation_stereo.tests.reporting import disparity_views


def test_disparity_shift_remains_visible_on_shared_scale():
    reference = np.linspace(0, 100, 700 * 700).reshape(700, 700)
    candidate = reference + 3
    views = disparity_views({"disparity": candidate}, {"disparity": reference})
    np.testing.assert_allclose(views["absolute_error"], 3)
    assert views["scale"] == (0.0, float(np.percentile(reference, 99)))
    np.testing.assert_array_equal(views["native"], candidate)
    np.testing.assert_array_equal(views["reference"], reference)
