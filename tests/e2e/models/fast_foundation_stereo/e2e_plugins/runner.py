# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Fast Foundation Stereo E2E runner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .contracts import E2ECase, RunContext, StageOutput, StageSpec


def _write_stereo_inputs(directory: Path, *, pixel_shift: int) -> tuple[Path, Path]:
    from PIL import Image

    if not 0 < pixel_shift < 700:
        raise ValueError(f"pixel_shift must be in [1, 699], got {pixel_shift}")
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:700, 0:700]
    left = np.stack(((x % 256), (y % 256), ((x + y) % 256)), axis=-1).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, :-pixel_shift] = left[:, pixel_shift:]
    right[:, -pixel_shift:] = left[:, -1:]
    left_path = directory / "left.png"
    right_path = directory / "right.png"
    Image.fromarray(left).save(left_path)
    Image.fromarray(right).save(right_path)
    return left_path, right_path


class StereoDisparityRunner:
    @property
    def strategy_name(self) -> str:
        return "stereo_disparity"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": f"Unknown stage: {stage.name}"},
            )
        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        pixel_shift = int(case.inputs.get("pixel_shift", 12))
        left_path, right_path = _write_stereo_inputs(
            artifact_dir,
            pixel_shift=pixel_shift,
        )
        output_path = artifact_dir / "disparity.f32"
        bundle = case.bundle or f"{case.name}.bundle"
        bundle_path = bundle if os.path.isabs(bundle) else os.path.join(ctx.engine_dir, bundle)
        command = [
            ctx.binary_path,
            "disparity",
            bundle_path,
            "--image",
            str(left_path),
            "--right-image",
            str(right_path),
            "--output",
            str(output_path),
            "--cuda-graphs",
        ]
        if ctx.model_plugin_dir:
            command.extend(("--model-plugin-dir", ctx.model_plugin_dir))
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode == 0 and output_path.is_file():
            disparity = np.fromfile(output_path, dtype=np.float32)
            if disparity.size == 700 * 700:
                data.update(
                    disparity=disparity.reshape(700, 700),
                    output_shape=[700, 700],
                    output_path=str(output_path),
                )
            try:
                data["cli_output"] = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": command},
        )


runner = StereoDisparityRunner()
