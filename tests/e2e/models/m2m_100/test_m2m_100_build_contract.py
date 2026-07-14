# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for M2M-100 and NLLB."""

from __future__ import annotations

import json
from pathlib import Path


def test_nllb_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "nllb-200.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
