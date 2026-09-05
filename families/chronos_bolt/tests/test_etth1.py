# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.chronos_bolt.tests.etth1 import GATES, _starts, windows


def test_etth1_selector_and_exact_gates() -> None:
    starts = _starts()
    assert starts == (13816, 13528, 12904, 11200, 12112, 13552, 11944, 12448, 12688, 13168)
    assert GATES == {"relative_l2": 1.0e-6, "max_pointwise_error": 8.0e-6}


def test_etth1_source_is_required(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_REFERENCE_SOURCE_DIR"):
        windows()
