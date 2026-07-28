# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for Bark."""

from __future__ import annotations

import json
from pathlib import Path


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_large_acceptance_build_preserves_audited_full_precision() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "bark-large.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"


def test_large_acceptance_build_has_a_precision_matched_l0() -> None:
    model_dir = Path(__file__).parent
    large = json.loads((model_dir / "manifests" / "bark-large.json").read_text(encoding="utf-8"))
    replacement = json.loads(
        (model_dir / "manifests" / "bark-small-fp32-l0.json").read_text(encoding="utf-8")
    )
    small = json.loads(
        (model_dir / "manifests" / "bark-small.json").read_text(encoding="utf-8")
    )
    small_threshold = json.loads(
        (model_dir / "thresholds" / "bark-small.json").read_text(encoding="utf-8")
    )
    replacement_threshold = json.loads(
        (model_dir / "thresholds" / "bark-small-fp32-l0.json").read_text(encoding="utf-8")
    )

    assert large["testcases"][0]["l0_replacement"] == replacement["name"]
    for field in ("family", "runtime_strategy", "precision", "quantization"):
        assert large.get(field) == replacement.get(field)
    assert replacement["hf_id"] == "suno/bark-small"
    assert replacement["bundle"] == "bark-small-fp32-l0.trtfb"
    assert replacement["e2e_parallel_resource"] == "exclusive_gpu"
    assert replacement["testcases"][0]["ci_tier"] == "l0_only"
    assert replacement["testcases"][0]["reference_precision"] == "fp32"
    for field in ("prompt", "max_new_tokens", "determinism"):
        assert replacement["testcases"][0].get(field) == small["testcases"][0].get(field)
    assert "build_env" not in replacement
    assert replacement_threshold == small_threshold


def test_acceptance_build_does_not_use_an_unqualified_timing_cache() -> None:
    model_dir = Path(__file__).parent
    manifest_path = model_dir / "manifests" / "bark-small.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp16"
    assert "build_env" not in manifest
