# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST-owned ETTh1 nightly windows."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

CASES = {
    "patchtst-etth1-regression-distribution": {
        "columns": ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"),
        "context": 512,
        "prediction": 1,
        "gates": {"relative_l2": 1.0e-3, "max_pointwise_error": 1.0e-3},
    },
    "patchtst-granite-official": {
        "columns": ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"),
        "context": 512,
        "prediction": 96,
        "gates": {"relative_l2": 1.5e-3, "max_pointwise_error": 3.5e-2},
    },
}


def _starts(context: int, prediction: int) -> tuple[int, ...]:
    starts = list(range(11520 - context, 14400 - context - prediction + 1, 24))
    random.Random(20260715).shuffle(starts)
    return tuple(starts[:10])


def windows(case_name: str) -> tuple[dict, ...]:
    config = CASES[case_name]
    root = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert root, "selected PatchTST ETTh1 E2E requires TRTMC_REFERENCE_SOURCE_DIR"
    path = Path(root) / "ETT-small/ETTh1.csv"
    assert path.is_file(), f"ETTh1 source is missing: {path}"
    columns = config["columns"]
    context = int(config["context"])
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert set(columns) <= set(reader.fieldnames or ())
        rows = list(reader)
    assert len(rows) >= 14400
    return tuple(
        {
            "past_values": [
                float(row[column]) for row in rows[start : start + context] for column in columns
            ]
        }
        for start in _starts(context, int(config["prediction"]))
    )


def gates(case_name: str) -> dict[str, float]:
    return dict(CASES[case_name]["gates"])
