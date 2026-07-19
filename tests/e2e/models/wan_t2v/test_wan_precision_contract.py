# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision and scheduling regression coverage for Wan's scale contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_MANIFEST_DIR = Path(__file__).parent / "manifests"


def test_nightly_wan_build_keeps_complete_t5_encoder_in_fp32() -> None:
    manifest = json.loads(
        (_MANIFEST_DIR / "wan21-t2v-1.3b.json").read_text()
    )

    assert manifest["precision"] == "fp16"
    assert manifest["fp32_layers"] == [24]


@pytest.mark.parametrize(
    "manifest_name",
    ("wan21-t2v-1.3b-l0.json", "wan21-t2v-1.3b.json"),
)
def test_single_device_wan_builds_reserve_an_exclusive_gpu(
    manifest_name: str,
) -> None:
    manifest = json.loads((_MANIFEST_DIR / manifest_name).read_text())

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
