# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion components check — component-by-component TRT vs HF comparison."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class DiffusionComponentsTest:
    name = "diffusion_components"
    description = "Diffusion pipeline: config, text encoder, denoiser, scheduler, full pipeline"
    runtime_strategies = []
    requires_bundle = True
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from debug_diffusion_pipeline import run_as_diff_test
        return run_as_diff_test(ctx)
