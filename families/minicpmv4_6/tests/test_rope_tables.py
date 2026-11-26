# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder's RoPE tables are finite for every cached position, or the build stops before any engine."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from families.minicpmv4_6.rope import rope_positions, rope_tables  # noqa: E402


def _text(theta):
    return SimpleNamespace(rotary_dim=64, rope_theta=theta)


def test_the_checkpoint_theta_gives_finite_tables():
    cos, sin = rope_tables(_text(1e7), rope_positions(4096))
    assert cos.shape == sin.shape == (4097, 32)
    assert np.isfinite(cos).all() and np.isfinite(sin).all()


@pytest.mark.parametrize("theta", [1e-38, 1e-40])
def test_a_theta_whose_fp32_frequencies_or_angles_overflow_is_rejected(theta):
    # 1e-38: the angles of late positions overflow; 1e-40: already the frequencies do.
    with pytest.raises(ValueError, match="overflows the FP32 RoPE angles"):
        rope_tables(_text(theta), rope_positions(4096))
