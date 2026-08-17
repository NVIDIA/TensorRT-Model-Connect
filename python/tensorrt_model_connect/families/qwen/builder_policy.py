# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder policy shared by Qwen decoder layouts."""

from __future__ import annotations

import os
from typing import Any


def configure_qwen_builder(
    trt_config: Any,
    quant_ctx: Any | None,
    trt_version: str,
) -> None:
    """Select the accuracy-stable tactic search level for Qwen FP8.

    TensorRT 11.2 and later 11.x releases can select numerically unstable FP8
    tactics for this graph.  Level 0 produces stable logits without adding
    runtime work.
    TensorRT 11.0 selects accurate tactics with its default search, so keep
    that behavior.  Preserve the process-wide compatibility override when a
    caller intentionally supplies one.
    """
    if "TRTMC_BUILDER_OPTIMIZATION_LEVEL" in os.environ:
        return

    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    version_parts = trt_version.split(".", maxsplit=2)
    if len(version_parts) < 2:
        return
    try:
        major_minor = tuple(int(part) for part in version_parts[:2])
    except ValueError:
        return

    if (
        getattr(quant_format, "name", None) == "fp8"
        and major_minor[0] == 11
        and major_minor[1] >= 2
    ):
        trt_config.builder_optimization_level = 0
