# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuTile kernel for the toy user kernel tests: ``out = a * b + c + 1`` over flat arrays.

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
