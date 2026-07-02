# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM model-owned E2E repro command provider."""

from __future__ import annotations

from .commands import build_sana_wm_trt_command
from .contracts import E2ECase, ReproCommandProvider, RunContext


class SanaWmReproCommandProvider:
    """Build SANA-WM TRT repro commands from the model-card manifest fields."""

    @property
    def family_name(self) -> str:
        return "sana_wm"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "diffusion_media_generation":
            return None
        return build_sana_wm_trt_command(
            case,
            ctx,
            bundle_path,
            "/tmp/trtmc_frames",
        )


repro_provider: ReproCommandProvider = SanaWmReproCommandProvider()
