# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.timesfm.tests.etth1 import GATES, _starts, windows


def test_etth1_selector_and_exact_gates() -> None:
    starts = _starts()
    assert starts == (12208, 10816, 11704, 9664, 10552, 11992, 10432, 11608, 11128, 11032)
    assert GATES == {"relative_l2": 4.0e-3, "max_pointwise_error": 7.0e-3}


def test_etth1_source_is_required(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_REFERENCE_SOURCE_DIR"):
        windows()
