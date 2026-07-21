# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned E2E repro command provider."""

from __future__ import annotations

import shlex
from pathlib import Path

from .contracts import E2ECase, ReproCommandProvider, RunContext
from .runners.diffusion import build_generate_video_command


class Wan22Ti2vReproCommandProvider:
    """Render the same native video command that the Wan2.2 runner executes."""

    @property
    def family_name(self) -> str:
        return "wan2_2_ti2v"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "diffusion_media_generation":
            return None

        command = build_generate_video_command(
            case,
            ctx,
            Path("/tmp/trtmc_wan22_ti2v_frames"),
            bundle_path=bundle_path,
        )
        prompt_index = command.index("--prompt") + 1
        command[prompt_index] = shlex.quote(command[prompt_index])
        return command


repro_provider: ReproCommandProvider = Wan22Ti2vReproCommandProvider()
