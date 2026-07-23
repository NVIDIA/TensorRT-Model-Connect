# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib

import numpy as np

from tensorrt_model_connect.families.openpi.numerics import (
    round_to_bfloat16_float32,
    sinusoidal_inverse_periods_numpy,
)


def test_bfloat16_rounding_is_ties_to_even_and_portably_stored_as_float32() -> None:
    source_bits = np.array([0x3F800000, 0x3F808000, 0x3F818000, 0xBF818000], dtype=np.uint32)
    rounded = round_to_bfloat16_float32(source_bits.view(np.float32))

    assert rounded.dtype == np.float32
    assert rounded.view(np.uint32).tolist() == [
        0x3F800000,
        0x3F800000,
        0x3F820000,
        0xBF820000,
    ]
    np.testing.assert_array_equal(round_to_bfloat16_float32(rounded), rounded)


def test_canonical_timestep_inverse_periods_match_pinned_xla_table() -> None:
    inverse_periods = sinusoidal_inverse_periods_numpy(1024)

    assert inverse_periods.shape == (512,)
    assert inverse_periods.dtype == np.float32
    assert hashlib.sha256(inverse_periods.astype("<f4").tobytes()).hexdigest() == (
        "1761ced44acfa477ba656be02ea923a0ab201cef26e43cafe63de22272890f03"
    )
