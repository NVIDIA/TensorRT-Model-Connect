# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python mirror of the family noise generator, which the E2E replays into the reference.

``tests/cpp/test_sampling.cpp`` checks the native generator against the same values.
"""

from __future__ import annotations

import numpy as np

from families.stable_audio.tests.family_noise import gaussian, initial_latents, mix64, step_noise


def test_splitmix_finalizer_matches_the_reference_constants():
    # SplitMix64 (Steele, Lea, Flood 2014) seeded with 0: first output after one increment.
    assert int(mix64(np.array([0x9E3779B97F4A7C15], dtype=np.uint64))[0]) == 0xE220A8397B1DCDAF


def test_values_are_the_native_golden_values():
    assert gaussian(0, 0, 6).tolist() == np.array(
        [-0.278874933719635, -0.8810646533966064, -0.7642228603363037, -1.1462910175323486,
         0.4189256429672241, 0.5488144159317017], dtype=np.float32).tolist()
    assert gaussian(123456789, 7, 5).tolist() == np.array(
        [1.4240012168884277, -0.8176755905151367, -0.5776025652885437, -0.015606191009283066,
         2.618398904800415], dtype=np.float32).tolist()


def test_streams_are_prefix_stable_and_independent():
    long = gaussian(5, 2, 1001)
    assert np.array_equal(gaussian(5, 2, 10), long[:10])
    assert not np.array_equal(gaussian(5, 3, 10), long[:10])
    assert not np.array_equal(gaussian(6, 2, 10), long[:10])


def test_values_are_standard_normal():
    values = gaussian(42, 1, 1 << 18).astype(np.float64)
    assert abs(values.mean()) < 0.01
    assert abs(values.std() - 1.0) < 0.01
    assert np.isfinite(values).all()


def test_latent_and_step_streams_have_the_latent_shape():
    assert initial_latents(3, 4, 5).shape == (1, 4, 5)
    assert np.array_equal(step_noise(3, 0, 4, 5).reshape(-1), gaussian(3, 1, 20))
