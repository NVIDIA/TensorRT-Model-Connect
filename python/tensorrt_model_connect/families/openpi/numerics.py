# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light numeric references shared by OpenPI builders and tests."""

from __future__ import annotations

import numpy as np


def sinusoidal_inverse_periods_numpy(
    dimension: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> np.ndarray:
    """Return inverse periods matching the pinned CUDA/XLA FP32 table."""
    if dimension <= 2 or dimension % 2:
        raise ValueError("OpenPI sinusoidal embedding dimension must be even and > 2")
    half_dimension = dimension // 2
    fraction = np.concatenate(
        [
            np.arange(half_dimension - 1, dtype=np.float32)
            * np.float32(1.0 / (half_dimension - 1)),
            np.ones(1, dtype=np.float32),
        ]
    )
    ratio = np.float32(max_period / min_period)
    period = np.float32(min_period) * np.power(ratio, fraction).astype(np.float32)
    inverse_periods = (np.float32(1.0) / period).astype(np.float32)
    if dimension != 1024 or min_period != 4e-3 or max_period != 4.0:
        return inverse_periods

    # CUDA's POW lowering in the pinned JAX/XLA oracle differs from NumPy's
    # host POW by at most two ULPs in 93 of 512 entries. Store those tiny
    # corrections once so inference does not execute 512 POW operations per
    # Euler step.
    corrections = np.zeros(half_dimension, dtype=np.int32)
    corrections[
        [
            12,
            18,
            28,
            32,
            77,
            107,
            113,
            116,
            127,
            131,
            134,
            148,
            162,
            171,
            174,
            181,
            183,
            195,
            200,
            212,
            220,
            227,
            256,
            262,
            274,
            277,
            281,
            282,
            284,
            290,
            297,
            302,
            303,
            304,
            333,
            358,
            360,
            364,
            388,
            399,
            409,
            412,
            420,
            435,
            448,
            452,
            475,
            476,
            480,
            487,
            491,
            495,
            503,
            506,
            507,
        ]
    ] = 1
    corrections[
        [
            3,
            17,
            20,
            34,
            40,
            49,
            64,
            76,
            111,
            118,
            215,
            222,
            230,
            244,
            251,
            252,
            259,
            289,
            307,
            314,
            326,
            357,
            389,
            393,
            406,
            417,
            445,
            460,
            463,
            464,
            478,
            482,
        ]
    ] = -1
    corrections[[110, 164, 264, 316, 325]] = 2
    corrections[238] = -2
    bits = inverse_periods.view(np.uint32)
    return (bits.astype(np.int64) + corrections).astype(np.uint32).view(np.float32)


def round_to_bfloat16_float32(value: np.ndarray) -> np.ndarray:
    """Return BF16-round-to-nearest-even values in portable FP32 storage.

    NumPy does not expose a portable bfloat16 dtype.  The official OpenPI
    loader nevertheless restores every policy parameter as BF16 before model
    execution, including parameters later consumed by FP32 layers.  Keeping
    the rounded values in FP32 lets TensorRT use those exact parameter values
    without adding a build- or runtime dependency on ``ml_dtypes``.
    """

    array = np.ascontiguousarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded_bits = (bits + rounding_bias) & np.uint32(0xFFFF0000)
    return np.ascontiguousarray(rounded_bits.view(np.float32))
