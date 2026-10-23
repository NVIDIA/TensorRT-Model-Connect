# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Gemma too large for a split pair must build one dual-profile plan instead.

The split layout keeps a separate decode engine with a static Sq=1 graph, which
costs a second full copy of the weights on the device. The qualification target
is an L40S at about 45 GiB, so a pair is only affordable up to roughly 19 GiB
per engine: gemma-3-12b needs 25 GiB each and its 50 GiB pair does not fit.
"""

from __future__ import annotations

import numpy as np

from families.gemma.model import (
    _MAX_SPLIT_ENGINE_BYTES,
    _decoder_engine_bytes,
)


def _weights(parameters: int) -> dict:
    # The estimator reads size only; preserve the shape without a huge backing array.
    return {"w": np.broadcast_to(np.zeros(1, dtype=np.float32), (parameters,))}


def test_bytes_follow_the_build_precision():
    half = _decoder_engine_bytes(_weights(1_000_000), "bf16")
    full = _decoder_engine_bytes(_weights(1_000_000), "fp32")

    assert full == 2 * half
    # 1e6 parameters at 2 bytes, plus the measured plan overhead.
    assert half == int(1_000_000 * 2 * 1.05)


def test_the_small_widths_stay_on_split():
    """Parameter counts of the shipped text decoders that still fit a pair."""
    for parameters in (0.27e9, 1.00e9, 2.61e9, 3.88e9):
        assert _decoder_engine_bytes(_weights(int(parameters)), "bf16") <= _MAX_SPLIT_ENGINE_BYTES


def test_12b_and_27b_exceed_the_split_budget():
    for parameters in (11.77e9, 27.01e9):
        assert _decoder_engine_bytes(_weights(int(parameters)), "bf16") > _MAX_SPLIT_ENGINE_BYTES


def test_a_pair_at_the_budget_fits_the_qualification_target():
    """Both engines plus working memory have to fit an L40S, not just the weights.

    The card reports 46068 MiB, so about 45 GiB; a pair at the budget is 38 GiB.
    """
    l40s_bytes = 46068 * 1024**2
    assert 2 * _MAX_SPLIT_ENGINE_BYTES < l40s_bytes
    assert l40s_bytes - 2 * _MAX_SPLIT_ENGINE_BYTES >= 5 * 1024**3
