"""Qwen Image model-owned E2E repro command provider."""

from __future__ import annotations

import shlex
from pathlib import Path

from . import _case_artifact_dir
from .contracts import E2ECase, ReproCommandProvider, RunContext


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


class QwenImageReproCommandProvider:
    """Build Qwen Image TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "qwen_image"

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
            "run",
            bundle_path,
            "--prompt",
            _shell_quote(case.inputs.get("prompt", case.inputs.get("test_prompt", ""))),
            "--output",
            "/tmp/trtmc_qwen_image/output.png",
            "--num-inference-steps",
            str(case.inputs.get("num_inference_steps", 20)),
        ]

        negative_prompt = case.inputs.get("negative_prompt")
        if negative_prompt is not None:
            infer_parts.extend(["--negative-prompt", _shell_quote(negative_prompt)])

        cfg_scale = case.inputs.get("cfg_scale")
        if cfg_scale is None:
            cfg_scale = case.inputs.get("guidance_scale")
        if cfg_scale is not None:
            infer_parts.extend(["--cfg-scale", str(cfg_scale)])

        height = case.inputs.get("height") or case.inputs.get("image_height")
        if height is not None:
            infer_parts.extend(["--height", str(height)])

        width = case.inputs.get("width") or case.inputs.get("image_width")
        if width is not None:
            infer_parts.extend(["--width", str(width)])

        if "seed" in case.inputs:
            infer_parts.extend(["--seed", str(case.inputs["seed"])])

        image_path = case.inputs.get("image") or case.inputs.get("image_path")
        if image_path:
            infer_parts.extend(["--image", _shell_quote(image_path)])

        if ctx.artifacts_dir:
            latent_path = Path(
                _case_artifact_dir(ctx.artifacts_dir, case.name)
            ) / "initial_latents.raw"
            infer_parts.extend(["--initial-latents-raw", str(latent_path)])

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])

        return infer_parts


repro_provider: ReproCommandProvider = QwenImageReproCommandProvider()
