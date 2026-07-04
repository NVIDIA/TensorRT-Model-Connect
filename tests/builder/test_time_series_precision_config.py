# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision configuration regressions for strongly typed time-series graphs."""

from pathlib import Path


def test_time_series_builders_do_not_use_removed_precision_flags() -> None:
    root = Path(__file__).resolve().parents[2]
    families = ("chronos_bolt", "patchtsmixer", "patchtst", "timesfm")

    for family in families:
        source = (
            root
            / "python"
            / "tensorrt_model_connect"
            / "families"
            / family
            / "time_series_trt.py"
        ).read_text(encoding="utf-8")
        assert "BuilderFlag.FP16" not in source, family
        assert "BuilderFlag.BF16" not in source, family
