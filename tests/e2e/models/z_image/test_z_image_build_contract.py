# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for Z-Image."""

from __future__ import annotations

import json
from pathlib import Path


def test_full_z_image_build_reserves_an_exclusive_gpu() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "z-image-turbo.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
