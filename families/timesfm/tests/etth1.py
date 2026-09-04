# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM-owned ETTh1 nightly windows."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

CASE = "timesfm-2.0-500m-official"
GATES = {"relative_l2": 4.0e-3, "max_pointwise_error": 7.0e-3}
_COLUMNS = ("OT",)
_CONTEXT = 2048
_PREDICTION = 128


def _starts() -> tuple[int, ...]:
    starts = list(range(11520 - _CONTEXT, 14400 - _CONTEXT - _PREDICTION + 1, 24))
    random.Random(20260715).shuffle(starts)
    return tuple(starts[:10])


def windows() -> tuple[dict, ...]:
    root = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert root, "selected TimesFM ETTh1 E2E requires TRTMC_REFERENCE_SOURCE_DIR"
    path = Path(root) / "ETT-small/ETTh1.csv"
    assert path.is_file(), f"ETTh1 source is missing: {path}"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert set(_COLUMNS) <= set(reader.fieldnames or ())
        rows = list(reader)
    assert len(rows) >= 14400
    return tuple(
        {
            "past_values": [
                float(row[column]) for row in rows[start : start + _CONTEXT] for column in _COLUMNS
            ],
            "frequency": 0,
        }
        for start in _starts()
    )
