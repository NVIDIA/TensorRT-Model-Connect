# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official NGC ONNX Runtime reference for FoundationPose."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import time

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


class FoundationPoseOnnxRuntimeReference:
    @property
    def backend_name(self) -> str:
        return "foundationpose_onnxruntime"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "synthetic_crop_pose_refinement":
            raise ValueError(f"Unsupported FoundationPose reference stage: {stage.name!r}")
        root = Path(os.environ.get("TRTMC_FOUNDATIONPOSE_MODEL_DIR", ""))
        refiner_path = root / "refine_model.onnx"
        scorer_path = root / "score_model.onnx"
        if not refiner_path.is_file() or not scorer_path.is_file():
            raise FileNotFoundError(
                "TRTMC_FOUNDATIONPOSE_MODEL_DIR must contain the pinned NGC pair"
            )
        fixture = Path(ctx.artifacts_dir or "/tmp") / case.name / "native"
        output_path = fixture.parent / "onnxruntime_reference.npz"
        output_path.unlink(missing_ok=True)
        script = Path(__file__).resolve().parents[1] / "official_reference.py"
        command = [
            ctx.reference_python_path() or sys.executable,
            str(script),
            "--model-dir",
            str(root),
            "--fixture-dir",
            str(fixture),
            "--output",
            str(output_path),
            "--num-hypotheses",
            str(case.inputs["num_hypotheses"]),
            "--refinement-iterations",
            str(case.inputs["refinement_iterations"]),
            "--mesh-diameter",
            str(case.inputs["mesh_diameter"]),
        ]
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, env=env)
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode == 0 and output_path.is_file():
            with np.load(output_path, allow_pickle=False) as payload:
                data.update(
                    refined_poses=np.array(payload["refined_poses"], copy=True),
                    scores=np.array(payload["scores"], copy=True),
                    best_index=int(payload["best_index"]),
                    output_path=str(output_path),
                )
        elif completed.returncode == 0:
            data["output_error"] = f"reference exited 0 but did not create {output_path}"
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": completed.returncode,
                "ngc_version": "1.0.1_onnx",
            },
        )


reference = FoundationPoseOnnxRuntimeReference()
