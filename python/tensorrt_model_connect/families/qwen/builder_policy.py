# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder policy shared by Qwen decoder layouts."""

from __future__ import annotations

import os
from typing import Any


def configure_qwen_builder(
    trt_config: Any,
    quant_ctx: Any | None,
) -> None:
    """Select the accuracy-stable tactic search level for Qwen FP8.

    TensorRT's default tactic search can select numerically unstable FP8
    tactics for this graph.  Level 0 produces stable logits without adding
    runtime work.  Preserve the process-wide compatibility override when a
    caller intentionally supplies one.
    """
    if "TRTMC_BUILDER_OPTIMIZATION_LEVEL" in os.environ:
        return

    quant_format = getattr(getattr(quant_ctx, "profile", None), "format", None)
    if getattr(quant_format, "name", None) == "fp8":
        trt_config.builder_optimization_level = 0
