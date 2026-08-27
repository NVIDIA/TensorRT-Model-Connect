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

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec
try:
    from .input_contract import attach_ground_truth, stage_stereo_inputs
except ImportError:  # Loaded as a top-level model plugin.
    from input_contract import attach_ground_truth, stage_stereo_inputs


def _write_stereo_inputs(directory: Path) -> tuple[Path, Path]:
    """Backward-compatible fixture helper used by the visual report tests."""
    return stage_stereo_inputs(directory, {})


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
        left_path, right_path = stage_stereo_inputs(artifact_dir, case.inputs)
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
                attach_ground_truth(data, case.inputs)
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
