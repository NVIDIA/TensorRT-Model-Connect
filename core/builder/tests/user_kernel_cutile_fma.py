# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuTile kernels for the toy user kernel tests: ``out = a * b + c + 1``.

``fma_tile`` works on flat arrays. ``fma_2d_tile`` works on 2-D arrays. ``fma_bcast_tile`` has the
second input broadcast along rows and the third along columns.

This module needs ``cuda.tile``. The tests import it only after checking that it is installed.
It has no ``from __future__ import annotations`` because cuTile reads the annotations.
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def fma_tile(a, b, c, out, TILE: ConstInt):
    i = ct.bid(0)
    ta = ct.load(a, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    tb = ct.load(b, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    tc = ct.load(c, index=(i,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(i,), tile=ta * tb + tc + 1.0)


@ct.kernel
def fma_2d_tile(a, b, c, out, TR: ConstInt, TC: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    ta = ct.load(a, index=(i, j), shape=(TR, TC), padding_mode=ct.PaddingMode.ZERO)
    tb = ct.load(b, index=(i, j), shape=(TR, TC), padding_mode=ct.PaddingMode.ZERO)
    tc = ct.load(c, index=(i, j), shape=(TR, TC), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(i, j), tile=ta * tb + tc + 1.0)


@ct.kernel
def fma_bcast_tile(x, s, r, out, TR: ConstInt, TC: ConstInt):
    """out[i, j] = x[i, j] * s[0, j] + r[i, 0] + 1, with s of shape (1, C) and r of shape (R, 1)."""
    i = ct.bid(0)
    j = ct.bid(1)
    tx = ct.load(x, index=(i, j), shape=(TR, TC), padding_mode=ct.PaddingMode.ZERO)
    ts = ct.load(s, index=(0, j), shape=(1, TC), padding_mode=ct.PaddingMode.ZERO)
    tr = ct.load(r, index=(i, 0), shape=(TR, 1), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(i, j), tile=tx * ts + tr + 1.0)
