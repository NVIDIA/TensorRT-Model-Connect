# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned FP8 projection-selection contracts."""

from __future__ import annotations

import fnmatch

from tensorrt_model_connect.families.qwen.plugin import QwenPlugin


def test_qwen_fp8_quantizes_only_up_projections() -> None:
    patterns = QwenPlugin().quant_exclude_patterns("fp8")
    projection_names = {
        (layer, projection): f"qwen/layer.{layer}.w_{projection}"
        for layer in range(28)
        for projection in ("q", "k", "v", "o", "gate", "up", "down")
    }

    selected = {
        layer_projection
        for layer_projection, weight_name in projection_names.items()
        if not any(
            fnmatch.fnmatch(weight_name, pattern)
            or fnmatch.fnmatch(weight_name.split("/", 1)[-1], pattern)
            for pattern in patterns
        )
    }

    assert selected == {
        (layer, "up")
        for layer in range(28)
    }
