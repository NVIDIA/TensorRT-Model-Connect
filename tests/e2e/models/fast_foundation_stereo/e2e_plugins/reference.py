# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official PyTorch reference for native stereo E2E coverage."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensorrt_model_connect.families.fast_foundation_stereo.prepare_model import (
    resolve_model_dir,
)

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def _checkpoint_snapshot(case: E2ECase, *, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            case.hf_id,
            revision=case.hf_revision or None,
            allow_patterns=["cfg.yaml", "model_best_bp2_serialize.pth"],
            local_files_only=local_files_only,
        )
    )


class FastFoundationStereoTorchReference:
    @property
    def backend_name(self) -> str:
        return "fast_foundation_stereo_torch"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown stage: {stage.name}"},
            )
        checkpoint = _checkpoint_snapshot(
            case,
            local_files_only=ctx.local_files_only,
        )
        staged = resolve_model_dir(
            checkpoint,
            local_files_only=ctx.local_files_only,
        )
        model_root = staged or checkpoint
        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        left_path = artifact_dir / "left.png"
        right_path = artifact_dir / "right.png"
        if not left_path.is_file() or not right_path.is_file():
            return StageOutput(
                stage_name=stage.name,
                data={"error": "Stereo input artifacts are missing"},
            )
        output_path = artifact_dir / "torch_disparity.npy"
        script = Path(__file__).resolve().parents[1] / "official_reference.py"
        command = [
            ctx.reference_python_path() or sys.executable,
            str(script),
            "--model-root",
            str(model_root),
            "--output",
            str(output_path),
            "--left-image",
            str(left_path),
            "--right-image",
            str(right_path),
            "--valid-iters",
            "8",
            "--max-disp",
            "192",
        ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode == 0 and output_path.is_file():
            disparity = np.load(output_path, allow_pickle=False)
            data.update(
                disparity=disparity,
                expected_shape=list(disparity.shape),
                requires_finite=True,
                requires_nonnegative=True,
                output_path=str(output_path),
            )
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": result.returncode,
                "stderr": result.stderr,
            },
        )


reference = FastFoundationStereoTorchReference()
