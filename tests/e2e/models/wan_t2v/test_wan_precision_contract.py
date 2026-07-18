# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision regression coverage for the full Wan nightly contract."""

from __future__ import annotations

import json
from pathlib import Path


def test_nightly_wan_build_keeps_complete_t5_encoder_in_fp32() -> None:
    manifest = json.loads(
        (Path(__file__).parent / "manifests" / "wan21-t2v-1.3b.json").read_text()
    )

    assert manifest["precision"] == "fp16"
    assert manifest["fp32_layers"] == [24]
