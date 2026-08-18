# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contract tests for TimesFM."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _builder_function_source(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[4]
    source_path = repo_root / "python/tensorrt_model_connect/families/timesfm/model.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_acceptance_build_reserves_gpu_for_stable_tactic_selection() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "timesfm-2.0-500m-official.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_acceptance_build_uses_fp32_for_point_forecast_parity() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "timesfm-2.0-500m-official.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert "fp32_layers" not in manifest


def test_acceptance_build_matches_hf_normalization_patch_selection() -> None:
    selector = _builder_function_source("_select_normalization_patch")
    builder = _builder_function_source("_build_timesfm_network")

    # HF selects the first patch with at least three real values and falls
    # back to the last patch only when none qualify. Starting with that
    # fallback and replacing it in reverse order implements all three cases:
    # full context, partial left padding, and fewer than three real values.
    assert "last_patch = num_patches - 1" in selector
    assert "range(last_patch - 1, -1, -1)" in selector
    assert "minimum_valid_exclusive = add_scalar(network, (1, 1, 1), 2.0)" in selector
    assert "trt.ElementWiseOperation.GREATER" in selector
    assert selector.count("network.add_select(") == 2
    assert "_select_normalization_patch(" in builder
