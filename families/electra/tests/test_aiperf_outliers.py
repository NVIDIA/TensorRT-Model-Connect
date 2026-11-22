# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's encoder-parity outlier allowance for its BF16 profile."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

AIPERF = Path(__file__).resolve().parent / "aiperf"


def test_rare_bf16_outliers_within_twice_the_tolerance_are_allowed() -> None:
    benchmark = yaml.safe_load((AIPERF / "electra-base-discriminator-bf16.yaml").read_text())[
        "benchmarks"
    ]["encoder_parity"]
    assert benchmark == {
        "gate": {
            "outliers": {
                "max_fraction": 0.01,
                "gate": {"min_cosine": 0.998, "max_relative_l2": 0.04},
            }
        }
    }
