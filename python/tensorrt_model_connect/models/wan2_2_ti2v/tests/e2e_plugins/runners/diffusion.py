# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native C++ ``generate-video`` runner for fixed-profile Wan2.2 TI2V-5B."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


def _bundle_path(case: E2ECase, ctx: RunContext) -> Path:
    bundle = Path(case.bundle)
    return bundle if bundle.is_absolute() else Path(ctx.engine_dir) / bundle


def _require_model_plugin_dir(ctx: RunContext) -> str:
    if not ctx.model_plugin_dir:
        raise ValueError("Wan2.2 TI2V E2E requires an explicit model_plugin_dir")
    return ctx.model_plugin_dir


def build_generate_video_command(
    case: E2ECase,
    ctx: RunContext,
    output_dir: Path,
    *,
    bundle_path: str | Path | None = None,
) -> list[str]:
    """Build the public native-runtime command used by the E2E stage."""

    bundle = Path(bundle_path) if bundle_path is not None else _bundle_path(case, ctx)
    model_plugin_dir = _require_model_plugin_dir(ctx)
    command = [
        ctx.binary_path,
        "generate-video",
        str(bundle),
        "--prompt",
        str(case.inputs.get("prompt", "")),
        "--output",
        str(output_dir),
        "--num-steps",
        str(case.inputs["num_inference_steps"]),
        "--guidance-scale",
        str(case.inputs["guidance_scale"]),
        "--seed",
        str(case.inputs["seed"]),
        "--height",
        str(case.inputs["video_height"]),
        "--width",
        str(case.inputs["video_width"]),
        "--backend-dir",
        str(Path(ctx.binary_path).parent),
    ]
    command.extend(["--model-plugin-dir", model_plugin_dir])
    return command


def _frame_stats(frame_paths: list[Path]) -> dict[str, float | int | bool]:
    """Compute exact aggregate statistics without retaining all 121 frames."""

    if not frame_paths:
        return {
            "mean": 0.0,
            "std": 0.0,
            "width": 0,
            "height": 0,
            "dimensions_consistent": False,
        }

    import numpy as np
    from PIL import Image

    total = 0.0
    total_squared = 0.0
    element_count = 0
    expected_size: tuple[int, int] | None = None
    dimensions_consistent = True
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            if expected_size is None:
                expected_size = rgb.size
            dimensions_consistent = dimensions_consistent and rgb.size == expected_size
            pixels = np.asarray(rgb, dtype=np.uint8)
        total += float(pixels.sum(dtype=np.float64))
        total_squared += float(np.square(pixels, dtype=np.float64).sum(dtype=np.float64))
        element_count += int(pixels.size)

    mean_u8 = total / element_count
    variance_u8 = max(total_squared / element_count - mean_u8 * mean_u8, 0.0)
    width, height = expected_size or (0, 0)
    return {
        "mean": mean_u8 / 255.0,
        "std": variance_u8**0.5 / 255.0,
        "width": width,
        "height": height,
        "dimensions_consistent": dimensions_consistent,
    }


class DiffusionMediaRunner:
    """Run only the public Python-free Model-Connect C++ video path."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != self.strategy_name:
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

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        generation_stages = {"end_to_end", "end_to_end_video", "generate", "frame_quality"}
        if stage.name not in generation_stages:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Wan2.2 TI2V stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )

        artifact_root = Path(ctx.artifacts_dir) / case.name if ctx.artifacts_dir else None
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)
        if artifact_root is None:
            output_dir = Path(tempfile.mkdtemp(prefix="wan22_ti2v_frames_"))
        else:
            output_dir = artifact_root / "frames"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        command = build_generate_video_command(case, ctx, output_dir)
        timeout_s = int(case.metadata.get("runtime_timeout_s", 14400))
        if timeout_s <= 0:
            raise ValueError("runtime_timeout_s must be positive")

        started = time.monotonic()
        model_plugin_dir = _require_model_plugin_dir(ctx)
        env = {
            **os.environ,
            "LD_LIBRARY_PATH": ctx.ld_library_path,
            "TRTMC_MODEL_PLUGIN_DIR": model_plugin_dir,
            "TRTMC_MODEL_PLUGIN_STRICT": "1",
        }
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=env,
        )
        elapsed = time.monotonic() - started
        frames = sorted(output_dir.glob("frame_*.png"))
        stats = _frame_stats(frames)
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir,
            "end_to_end",
            case.name,
        )
        data = {
            "returncode": result.returncode,
            "num_frames": len(frames),
            "frames_dir": str(output_dir),
            "frame_paths": [str(path) for path in frames],
            "frame_stats": stats,
        }
        metadata = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr,
        }
        if stderr_log:
            metadata["stderr_log"] = stderr_log
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata=metadata,
        )


plugin = DiffusionMediaRunner()
