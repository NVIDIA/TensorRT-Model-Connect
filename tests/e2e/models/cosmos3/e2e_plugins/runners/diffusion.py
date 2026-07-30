# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native distributed ``generate-video`` runner for Cosmos3-Nano."""

from __future__ import annotations

import os
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
        raise ValueError("Cosmos3 E2E requires an explicit model_plugin_dir")
    return ctx.model_plugin_dir


def _distributed_config(case: E2ECase) -> dict:
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _prepare_distributed_env(case: E2ECase, ctx: RunContext, env: dict[str, str]) -> None:
    config = _distributed_config(case)
    if not config or env.get("TRTMC_NCCL_RENDEZVOUS"):
        return
    root = Path(ctx.artifacts_dir) / case.name if ctx.artifacts_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    rendezvous = root / "cosmos3_cp.nccl_rendezvous.bin"
    try:
        rendezvous.unlink()
    except FileNotFoundError:
        pass
    env["TRTMC_NCCL_RENDEZVOUS"] = str(rendezvous)


def _wrap_distributed(command: list[str], case: E2ECase, env: dict[str, str]) -> list[str]:
    config = _distributed_config(case)
    if not config:
        return command
    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", 1) or 1)
    wrapped = [launcher, "--tag-output", "-np", str(world_size)]
    export_env = config.get("export_env", [])
    if isinstance(export_env, list) and Path(launcher).name == "mpirun":
        for name in (str(item) for item in export_env):
            if name in env:
                wrapped.extend(["-x", name])
    return wrapped + command


def build_generate_video_command(
    case: E2ECase,
    ctx: RunContext,
    output_dir: Path,
    *,
    bundle_path: str | Path | None = None,
) -> list[str]:
    bundle = Path(bundle_path) if bundle_path is not None else _bundle_path(case, ctx)
    return [
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
        "--model-plugin-dir",
        _require_model_plugin_dir(ctx),
    ]


def _frame_stats(frame_paths: list[Path]) -> dict[str, float | int | bool]:
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
        env = {**os.environ, "LD_LIBRARY_PATH": ctx.ld_library_path}
        _prepare_distributed_env(case, ctx, env)
        command = build_generate_video_command(
            case,
            ctx,
            Path("/tmp/trtmc_cosmos3_frames"),
            bundle_path=bundle_path,
        )
        return _wrap_distributed(command, case, env)

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "end_to_end":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Cosmos3 stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        artifact_root = Path(ctx.artifacts_dir) / case.name if ctx.artifacts_dir else None
        if artifact_root is None:
            output_dir = Path(tempfile.mkdtemp(prefix="cosmos3_frames_"))
        else:
            artifact_root.mkdir(parents=True, exist_ok=True)
            output_dir = artifact_root / "frames"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        command = build_generate_video_command(case, ctx, output_dir)
        env = {
            **os.environ,
            "LD_LIBRARY_PATH": ctx.ld_library_path,
            "TRTMC_MODEL_PLUGIN_DIR": _require_model_plugin_dir(ctx),
            "TRTMC_MODEL_PLUGIN_STRICT": "1",
        }
        _prepare_distributed_env(case, ctx, env)
        command = _wrap_distributed(command, case, env)
        timeout_s = int(case.metadata.get("runtime_timeout_s", 3600))
        if timeout_s <= 0:
            raise ValueError("runtime_timeout_s must be positive")
        started = time.monotonic()
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
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir,
            stage.name,
            case.name,
        )
        data = {
            "returncode": result.returncode,
            "num_frames": len(frames),
            "frames_dir": str(output_dir),
            "frame_paths": [str(path) for path in frames],
            "frame_stats": _frame_stats(frames),
        }
        metadata = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr,
            "distributed_runtime": _distributed_config(case),
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
