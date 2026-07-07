# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM model-owned HF diffusers reference plugins."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import _case_artifact_dir
from .commands import build_sana_wm_reference_command
from .contracts import E2ECase, RunContext, StageOutput, StageSpec


def _ref_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    cublas = "/usr/local/cuda/lib64/libcublas.so.13"
    cublas_lt = "/usr/local/cuda/lib64/libcublasLt.so.13"
    if os.path.exists(cublas) and os.path.exists(cublas_lt):
        existing = env.get("LD_PRELOAD", "")
        preload = f"{cublas}:{cublas_lt}"
        env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


class SanaWmHfDiffusersReference:
    """SANA-WM local reference for hf_diffusers."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name not in {
            "end_to_end",
            "end_to_end_video",
            "generate",
            "frame_quality",
        }:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported SANA-WM reference stage: {stage.name}"},
            )
        return self._run_full_pipeline(case, stage, ctx)

    def _run_full_pipeline(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        python = ctx.reference_python_path() or sys.executable
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = (
            _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        )
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)
        cmd = build_sana_wm_reference_command(case, python, frames_dir)

        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[5],
            capture_output=True,
            text=True,
            timeout=7200,
            env=_ref_subprocess_env(),
        )
        elapsed = time.monotonic() - t0
        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        return StageOutput(
            stage_name=stage.name,
            data={
                "returncode": result.returncode,
                "num_frames": len(frame_files),
                "frames_dir": frames_dir,
                "frame_stats": self._compute_frame_stats(frames_dir),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers", "command": cmd},
        )

    @staticmethod
    def _compute_frame_stats(frame_dir: str) -> dict:
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return {"error": "PIL or numpy not available"}

        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frames:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        pixels = []
        for frame in frames:
            arr = np.array(Image.open(frame).convert("RGB"), dtype=np.float32) / 255.0
            pixels.append(arr.flatten())
        combined = np.concatenate(pixels)
        return {
            "count": len(frames),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
        }


reference = SanaWmHfDiffusersReference()
