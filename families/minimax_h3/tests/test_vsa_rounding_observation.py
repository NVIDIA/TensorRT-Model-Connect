# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct BF16 nearest-even rounding must not first round an FP64 midpoint."""

import math

import pytest

from .reference_vsa_rounding import nearest_bf16


@pytest.mark.parametrize("sign", [-1, 1])
def test_values_around_a_midpoint_survive_FP32_double_rounding(sign):
    lower, upper = 1.0, 1.0078125
    midpoint = (lower + upper) / 2
    assert nearest_bf16(sign * (midpoint - 2**-30)) == sign * lower
    assert nearest_bf16(sign * (midpoint + 2**-30)) == sign * upper
    assert nearest_bf16(sign * midpoint) == sign * lower
    lower, upper = upper, 1.015625
    midpoint = (lower + upper) / 2
    assert nearest_bf16(sign * (midpoint - 2**-30)) == sign * lower
    assert nearest_bf16(sign * midpoint) == sign * upper


def test_zero_sign_and_small_finite_values():
    assert math.copysign(1, nearest_bf16(-0.0)) == -1
    assert nearest_bf16(0.0) == 0
    assert nearest_bf16(2**-133) == 2**-133
    assert nearest_bf16(-(2**-133)) == -(2**-133)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e100])
def test_nonfinite_or_overflowing_diagnostics_are_rejected(value):
    with pytest.raises(ValueError):
        nearest_bf16(value)
