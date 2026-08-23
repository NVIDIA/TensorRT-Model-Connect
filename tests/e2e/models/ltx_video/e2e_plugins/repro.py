# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video model-owned E2E repro command provider."""

from __future__ import annotations

from pathlib import Path

from . import _case_artifact_dir
from .contracts import E2ECase, ReproCommandProvider, RunContext


class LtxVideoReproCommandProvider:
    """Build LTX Video TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "ltx_video"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "diffusion_media_generation":
            return None

        infer_parts = [
            ctx.binary_path,
            "generate-video",
            bundle_path,
            "--prompt",
            str(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
            "--output",
            "/tmp/trtmc_frames",
            "--num-steps",
            str(case.inputs.get("num_inference_steps", 30)),
        ]

        guidance_scale = case.inputs.get("guidance_scale")
        if guidance_scale is not None:
            infer_parts.extend(["--guidance-scale", str(guidance_scale)])

        if "seed" in case.inputs:
            infer_parts.extend(["--seed", str(case.inputs["seed"])])

        if ctx.artifacts_dir:
            latent_path = Path(
                _case_artifact_dir(ctx.artifacts_dir, case.name)
            ) / "initial_latents.raw"
            infer_parts.extend(["--initial-latents-raw", str(latent_path)])

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])

        return infer_parts


repro_provider: ReproCommandProvider = LtxVideoReproCommandProvider()
