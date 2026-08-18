# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Float32 libm operations matching the supported SAM2 source bit-for-bit."""

from __future__ import annotations

import ctypes

import numpy as np


_LIBM = ctypes.CDLL("libm.so.6")


def _unary(name: str):
    function = getattr(_LIBM, name)
    function.argtypes = (ctypes.c_float,)
    function.restype = ctypes.c_float
    return function


_SINF = _unary("sinf")
_COSF = _unary("cosf")
_SQRTF = _unary("sqrtf")
_POWF = _LIBM.powf
_POWF.argtypes = (ctypes.c_float, ctypes.c_float)
_POWF.restype = ctypes.c_float


def sinf(value: float) -> float:
    return float(_SINF(value))


def cosf(value: float) -> float:
    return float(_COSF(value))


def powf(base: float, exponent: float) -> float:
    return float(_POWF(base, exponent))


def reciprocal_sqrtf(value: float) -> float:
    root = np.float32(_SQRTF(value))
    return float(np.float32(np.float32(1.0) / root))
