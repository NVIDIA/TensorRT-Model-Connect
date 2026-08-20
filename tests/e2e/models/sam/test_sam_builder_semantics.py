# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static semantic checks for the SAM TensorRT graph builder."""

from __future__ import annotations

import ast
from pathlib import Path


SAM_PLUGIN = (
    Path(__file__).parents[4]
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "sam"
    / "plugin.py"
)


def test_sam_vision_builder_preserves_window_attention() -> None:
    """Local SAM layers must retain their trained window-attention scope."""

    source = SAM_PLUGIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    window_builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_windowed_attention"
    )
    segment = ast.get_source_segment(source, window_builder) or ""

    assert "_build_spatial_attention" in segment
    assert "spatial_size=window_size" in segment
    assert "window_count * window_count" in segment
    assert "treat windowed attention as global" not in segment.lower()
