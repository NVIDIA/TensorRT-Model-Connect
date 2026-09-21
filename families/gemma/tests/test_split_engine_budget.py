# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Gemma too large for a split pair must build one dual-profile plan instead.

The split layout keeps a separate decode engine with a static Sq=1 graph, which
costs a second full copy of the weights on the device. gemma-3-27b at bf16 needs
about 50 GiB per engine, so the pair cannot be deserialized on an 80 GiB device.
"""

from __future__ import annotations

import numpy as np

from families.gemma.model import (
    _MAX_SPLIT_ENGINE_BYTES,
    _decoder_engine_bytes,
)


def _weights(parameters: int) -> dict:
    return {"w": np.zeros(parameters, dtype=np.float32)}


def test_bytes_follow_the_build_precision():
    half = _decoder_engine_bytes(_weights(1_000_000), "bf16")
    full = _decoder_engine_bytes(_weights(1_000_000), "fp32")

    assert full == 2 * half
    # 1e6 parameters at 2 bytes, plus the measured plan overhead.
    assert half == int(1_000_000 * 2 * 1.05)


def test_the_qualified_widths_stay_on_split():
    # Parameter counts of the shipped text decoders.
    for parameters in (0.27e9, 1.00e9, 2.61e9, 3.88e9, 11.77e9):
        assert _decoder_engine_bytes(_weights(int(parameters)), "bf16") <= _MAX_SPLIT_ENGINE_BYTES


def test_27b_exceeds_the_split_budget():
    assert _decoder_engine_bytes(_weights(int(27.01e9)), "bf16") > _MAX_SPLIT_ENGINE_BYTES


def test_the_budget_leaves_room_for_a_pair_on_an_80_gib_device():
    """Both engines plus working memory have to fit, not just the weights."""
    assert 2 * _MAX_SPLIT_ENGINE_BYTES < 80 * 1024**3
