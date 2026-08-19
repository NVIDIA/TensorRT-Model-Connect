# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for Mixtral."""

from __future__ import annotations

import json
from pathlib import Path


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "mixtral-stories-15m.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_mmlu_continuation_build_stabilizes_first_divergence() -> None:
    """Keep the layers required for exact mmlu_000003 token-33 parity."""
    manifest_path = Path(__file__).parent / "manifests" / "mixtral-stories-15m.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp16"
    assert manifest["fp32_layers"] == [3, 4, 5]
