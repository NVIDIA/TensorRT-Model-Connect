# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer diff check — per-layer hidden state comparison: TRT vs HF."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class LayerDiffTest:
    name = "layer_diff"
    description = "Per-layer hidden state comparison: TRT vs HF transformers"
    requires_bundle = False
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from diff_layers import run_as_diff_test
        return run_as_diff_test(ctx)
