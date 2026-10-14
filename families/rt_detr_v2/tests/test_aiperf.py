# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]


def test_rtdetr_benchmark_uses_the_fixed_bundle_threshold():
    models = pytest.importorskip("trtmc_aiperf_qual.models")
    from trtmc_aiperf_qual.config import Environment

    resolve_model = models.resolve_model
    model = resolve_model("rtdetr-v2-r18", Environment({"repo": str(REPOSITORY)}))
    entry, = model["absolute"]
    assert entry["suite"] == "coco-map"
    assert entry["suite_definition"]["request"] == {"score_threshold": 0.3}
    assert entry["suite_definition"]["selection"]["count"] == 500
    assert entry["gate"] == {"margin": 1.0}
