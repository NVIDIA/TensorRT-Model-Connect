# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for Z-Image."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_MANIFEST_DIR = Path(__file__).parent / "manifests"


@pytest.mark.parametrize(
    "manifest_name",
    (
        "z-image-turbo-l0.json",
        "z-image-turbo.json",
        "z-image-turbo-l0-tp2.json",
    ),
)
def test_z_image_builds_reserve_an_exclusive_gpu(manifest_name: str) -> None:
    manifest = json.loads(
        (_MANIFEST_DIR / manifest_name).read_text(encoding="utf-8")
    )

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
