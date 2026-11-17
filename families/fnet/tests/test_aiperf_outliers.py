# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's encoder-parity outlier allowance for its FP16 profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PROFILES = ("fnet-base",)
AIPERF = Path(__file__).resolve().parent / "aiperf"


@pytest.mark.parametrize("profile", PROFILES)
def test_rare_fp16_outliers_within_twice_the_tolerance_are_allowed(profile: str) -> None:
    benchmark = yaml.safe_load((AIPERF / f"{profile}.yaml").read_text())["benchmarks"][
        "encoder_parity"
    ]
    assert benchmark["gate"] == {
        "outliers": {"max_fraction": 0.01, "gate": {"min_cosine": 0.998, "max_relative_l2": 0.04}}
    }
    assert benchmark["mismatched_precision_gate"] == {
        "outliers": {"max_fraction": 0.01, "gate": {"min_cosine": 0.996, "max_relative_l2": 0.08}}
    }
