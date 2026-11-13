# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]


def test_birefnet_owns_its_binary_conversion_benchmark_and_native_loader(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY / "apps/aiperf_qual"))
    from trtmc_aiperf_qual.config import Environment
    from trtmc_aiperf_qual.models import resolve_model

    model = resolve_model("birefnet-lite", Environment({"repo": str(REPOSITORY)}))
    entry, = model["absolute"]
    assert entry["metric"] == "mask_parity"
    assert entry["gate"] == {"min_mask_iou": 0.9}
    suite = entry["suite_definition"]
    assert suite["source"]["dataset"] == "detection-datasets/coco"
    assert suite["source"]["revision"] == "cf0b22332314a937e9dc8a1957b21725430bb41d"
    assert suite["selection"] == {"method": "seeded", "count": 50}
    assert "label_column" not in suite["source"] and not suite.get("request")
    assert model["reference"]["adapter"] == "families/birefnet/reference/adapter.py"
    assert model["reference"]["trust_remote_code"] is True
