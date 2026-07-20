# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the SANA-WM TensorRT builders."""

from pathlib import Path

import pytest


FAMILY_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "sana_wm"
)


@pytest.mark.parametrize(
    "relative_path",
    [
        "stage1_dit_builder.py",
        "refiner_dit_builder.py",
        "refiner_text_connector_builder.py",
    ],
)
def test_production_builders_do_not_suppress_tensorrt_search(relative_path: str) -> None:
    source = (FAMILY_DIR / relative_path).read_text(encoding="utf-8")

    forbidden_assignments = (
        "builder_optimization_level = 0",
        "max_num_tactics = 1",
        "tiling_optimization_level = trt.TilingOptimizationLevel.NONE",
    )
    for assignment in forbidden_assignments:
        assert assignment not in source
