# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for Llama models."""

from __future__ import annotations

import json
from pathlib import Path


def test_falcon3_split_decoder_build_reserves_an_exclusive_gpu() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "falcon3-1b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
