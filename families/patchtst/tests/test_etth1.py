# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.patchtst.tests.etth1 import _starts, gates, windows


def test_etth1_selectors_and_exact_gates() -> None:
    assert _starts(512, 1) == (
        12928,
        13768,
        13648,
        11200,
        13408,
        12568,
        13312,
        13864,
        11968,
        13144,
    )
    assert _starts(512, 96) == (
        13792,
        13528,
        12904,
        11200,
        12112,
        13552,
        11944,
        12448,
        12688,
        13168,
    )
    assert gates("patchtst-etth1-regression-distribution") == {
        "relative_l2": 1.0e-3,
        "max_pointwise_error": 1.0e-3,
    }
    assert gates("patchtst-granite-official") == {
        "relative_l2": 1.5e-3,
        "max_pointwise_error": 3.5e-2,
    }


def test_etth1_source_is_required(monkeypatch) -> None:
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_REFERENCE_SOURCE_DIR"):
        windows("patchtst-granite-official")
