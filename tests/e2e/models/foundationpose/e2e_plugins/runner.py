# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FoundationPose qualification runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


class FoundationPoseRunner:
    @property
    def strategy_name(self) -> str:
        return "pose_hypothesis_refinement"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "synthetic_crop_pose_refinement":
            raise ValueError(f"Unsupported FoundationPose runtime stage: {stage.name!r}")
        directory = Path(ctx.artifacts_dir or "/tmp") / case.name / "native"
        directory.mkdir(parents=True, exist_ok=True)
        executable = Path(ctx.binary_path).with_name("test_foundationpose_pipeline")
        bundle = Path(case.bundle)
        if not bundle.is_absolute():
            bundle = Path(ctx.engine_dir) / bundle
        command = [
            str(executable), "--qualify", "--bundle", str(bundle),
            "--output-dir", str(directory), "--backend-dir", str(Path(ctx.binary_path).parent),
            "--benchmark", "20", "--warmup", "3",
            "--num-hypotheses", str(case.inputs["num_hypotheses"]),
            "--refinement-iterations", str(case.inputs["refinement_iterations"]),
            "--mesh-diameter", str(case.inputs["mesh_diameter"]),
        ]
        if ctx.model_plugin_dir:
            command.extend(("--model-plugin-dir", ctx.model_plugin_dir))
        environment = dict(os.environ)
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=environment)
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode == 0:
            try:
                data.update(
                    summary=json.loads(completed.stdout),
                    refined_poses=np.fromfile(
                        directory / "trt_refined_poses.f32", dtype="<f4"
                    ).reshape(int(case.inputs["num_hypotheses"]), 4, 4),
                    scores=np.fromfile(directory / "trt_scores.f32", dtype="<f4"),
                    fixture_dir=str(directory),
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                data["output_error"] = str(error)
        return StageOutput(stage_name=stage.name, data=data, timing_s=elapsed,
                           metadata={"command": command, "returncode": completed.returncode})


runner = FoundationPoseRunner()
