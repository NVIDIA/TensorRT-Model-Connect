# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native MoGe monocular-geometry E2E runner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def _bundle_path(case: E2ECase, context: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.bundle"
    return bundle if os.path.isabs(bundle) else os.path.join(context.engine_dir, bundle)


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f"{label} has {array.size} values, expected shape {shape}")
    return array.reshape(shape)


def _load_geometry_output(output_dir: Path, cli_output: dict) -> dict:
    height = int(cli_output.get("height", 0))
    width = int(cli_output.get("width", 0))
    if height <= 0 or width <= 0:
        raise ValueError("geometry CLI reported invalid output dimensions")

    points_path = output_dir / "points.f32"
    depth_path = output_dir / "depth.f32"
    mask_path = output_dir / "mask.u8"
    intrinsics_path = output_dir / "intrinsics.json"
    for path in (points_path, depth_path, mask_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(f"geometry CLI did not create {path}")

    points = _require_shape(
        np.fromfile(points_path, dtype="<f4"), (height, width, 3), "points"
    )
    depth = _require_shape(np.fromfile(depth_path, dtype="<f4"), (height, width), "depth")
    mask = _require_shape(np.fromfile(mask_path, dtype=np.uint8), (height, width), "mask")
    intrinsics_payload = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(intrinsics_payload.get("intrinsics"), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics has invalid shape {intrinsics.shape}")
    if intrinsics_payload.get("normalized") is not True:
        raise ValueError("intrinsics output is not marked normalized")

    return {
        "points": points,
        "depth": depth,
        "mask": mask,
        "intrinsics": intrinsics,
        "height": height,
        "width": width,
        "num_tokens": 1800,
        "cli_output": cli_output,
        "points_path": str(points_path),
        "depth_path": str(depth_path),
        "mask_path": str(mask_path),
        "intrinsics_path": str(intrinsics_path),
    }


class MonocularGeometryRunner:
    @property
    def strategy_name(self) -> str:
        return "monocular_geometry"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported MoGe runtime stage: {stage.name!r}")
        num_tokens = int(case.inputs.get("num_tokens", 0))
        if num_tokens != 1800:
            raise ValueError(f"MoGe E2E requires num_tokens=1800, got {num_tokens}")
        image = Path(str(case.inputs.get("image", "")))
        if not image.is_file():
            raise FileNotFoundError(f"MoGe E2E input image is missing: {image}")

        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        output_dir = artifact_dir / "trt_geometry"
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("points.f32", "depth.f32", "mask.u8", "intrinsics.json"):
            (output_dir / filename).unlink(missing_ok=True)
        command = [
            ctx.binary_path,
            "geometry",
            _bundle_path(case, ctx),
            "--image",
            str(image),
            "--output",
            str(output_dir),
        ]
        if ctx.model_plugin_dir:
            command.extend(("--model-plugin-dir", ctx.model_plugin_dir))
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "num_tokens": num_tokens,
        }
        if completed.returncode == 0:
            try:
                data.update(_load_geometry_output(output_dir, json.loads(completed.stdout)))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                data["output_error"] = str(error)
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": command, "returncode": completed.returncode},
        )


runner = MonocularGeometryRunner()
