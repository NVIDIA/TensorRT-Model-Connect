# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM normalization-patch selection contract."""

from __future__ import annotations

import ast
from pathlib import Path


def _builder_function_source(name: str) -> str:
    source_path = Path(__file__).resolve().parents[1] / "model.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_build_matches_hf_normalization_patch_selection() -> None:
    selector = _builder_function_source("_select_normalization_patch")
    builder = _builder_function_source("_build_timesfm_network")

    assert "last_patch = num_patches - 1" in selector
    assert "range(last_patch - 1, -1, -1)" in selector
    assert "minimum_valid_exclusive = add_scalar(network, (1, 1, 1), 2.0)" in selector
    assert "trt.ElementWiseOperation.GREATER" in selector
    assert selector.count("network.add_select(") == 2
    assert "_select_normalization_patch(" in builder
