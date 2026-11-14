# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The NumPy twin of the runtime sampling arithmetic agrees with the C++ values.

``tests/cpp/test_sampling.cpp`` asserts the same golden values on the native side,
so both implementations are pinned to one definition.
"""

import numpy as np
import pytest

from families.stable_audio_3.tests import family_sampling as fs

# runtime/sampling.h gaussian_noise(42, draw, 7).
NOISE_42_DRAW_0 = [-0.318013877, -1.34076893, 0.258929193, 0.470199823, 1.16252971, -0.672291458, -0.419299811]
NOISE_42_DRAW_3 = [1.17811048, -0.363997906, 1.00697529, 1.75008237, -0.573026538, -0.352358818, 1.22086978]
# schedule(8, 108, LogSnrShift{}) and its half-precision rounding.
SCHEDULE_8 = [1.0, 0.994375587, 0.984480262, 0.957912266, 0.890903175, 0.745546579, 0.512497365, 0.273885012, 0.0]
SCHEDULE_8_HALF = [1.0, 0.994140625, 0.984375, 0.958007812, 0.891113281, 0.745605469, 0.512695312, 0.273925781, 0.0]


def test_noise_matches_the_native_stream():
    np.testing.assert_allclose(fs.gaussian_noise(42, 0, 7), NOISE_42_DRAW_0, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fs.gaussian_noise(42, 3, 7), NOISE_42_DRAW_3, rtol=1e-6, atol=1e-7)


def test_noise_streams_are_prefix_stable_and_distinct():
    long = fs.gaussian_noise(7, 1, 1001)
    np.testing.assert_array_equal(long[:10], fs.gaussian_noise(7, 1, 10))
    assert not np.array_equal(long[:10], fs.gaussian_noise(7, 2, 10))
    assert abs(float(long.mean())) < 0.15 and 0.85 < float(long.std()) < 1.15


def test_schedule_matches_the_reference_logsnr_shift():
    np.testing.assert_allclose(fs.schedule(8, 108), SCHEDULE_8, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(fs.round_to_half(fs.schedule(8, 108)), np.float32(SCHEDULE_8_HALF))


def test_linspace_is_torch_two_sided():
    values = fs.linspace_f32(1.0, 0.0, 9)
    assert values[0] == 1.0 and values[-1] == 0.0
    np.testing.assert_allclose(values, np.linspace(1.0, 0.0, 9), atol=1e-7)


@pytest.mark.parametrize(
    "seconds, latents, effective, valid, samples",
    [
        (0.5, 70, 6, 70, 22050),
        (10.0, 174, 108, 172, 441000),
        (47.3, 574, 510, 574, 2085929),
        (114.0, 1292, 1228, 1292, 5027400),
        (120.0, 1292, 1292, 1292, 5292000),
        (300.0, 1292, 3230, 1292, 5292032),
        (3.0e8, 1292, 3229980469, 1292, 5292032),
    ],
)
def test_render_plan_follows_generate(seconds, latents, effective, valid, samples):
    assert fs.plan_render(seconds) == fs.RenderPlan(latents, effective, valid, samples)


def test_render_plan_rejects_non_positive_durations():
    for seconds in (0.0, -1.0, float("nan"), float("inf"), 1.0e9 * 1.0000001):
        with pytest.raises(ValueError):
            fs.plan_render(seconds)
