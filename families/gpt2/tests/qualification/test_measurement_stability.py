# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.gpt2.tests.qualification.executor import measurement_stability


def test_historical_stability_requires_both_drift_and_median_band():
    assert measurement_stability([100.0] * 8 + [90.0, 110.0])["stable"]
    drift = measurement_stability([100.0] * 5 + [106.0] * 5)
    assert drift["samples_within_five_percent"] == 10
    assert not drift["stable"]
    spread = measurement_stability([90.0, 100.0, 110.0, 100.0, 100.0] * 2)
    assert spread["median_drift"] == 0.0
    assert not spread["stable"]


@pytest.mark.parametrize(
    "samples", [[1.0] * 20, [1.0] * 9, [1.0] * 9 + [float("nan")], [1.0] * 9 + [0.0]]
)
def test_invalid_measurements_cannot_pass_stability(samples):
    with pytest.raises(ValueError, match="ten finite positive"):
        measurement_stability(samples)
