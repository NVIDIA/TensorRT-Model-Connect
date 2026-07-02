# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM model-owned E2E runner plugins."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import _case_artifact_dir, save_full_stderr
from .commands import build_sana_wm_trt_command
from .contracts import E2ECase, RunContext, StageOutput, StageSpec


def _build_ld_library_path(ctx: RunContext) -> str:
    if ctx.ld_library_path:
        return ctx.ld_library_path
    parts: list[str] = []
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec and spec.submodule_search_locations:
        parts.append(spec.submodule_search_locations[0])
    parts.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle_name = case.bundle or case.inputs.get("bundle", "") or f"{case.name}.trtfb"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


class SanaWmDiffusionMediaGenerationRunner:
    """SANA-WM local runner for diffusion_media_generation."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

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
                data={"error": f"Unsupported SANA-WM stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return self._run_end_to_end(case, stage, ctx)

    def _run_end_to_end(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        bundle_path = _resolve_bundle_path(case, ctx)
        ld_path = _build_ld_library_path(ctx)

        with tempfile.TemporaryDirectory(prefix="trtmc_sana_wm_frames_") as frame_dir:
            cmd = build_sana_wm_trt_command(case, ctx, bundle_path, frame_dir)
            t0 = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                env={**os.environ, "LD_LIBRARY_PATH": ld_path},
            )
            elapsed = time.monotonic() - t0

            frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
            num_frames = len(frame_files)
            frame_stats = self._compute_frame_stats(frame_dir) if num_frames else {}
            frame_paths = [str(path) for path in frame_files]

            artifact_frames_dir = None
            if ctx.artifacts_dir and num_frames > 0:
                artifact_frames_dir = os.path.join(
                    _case_artifact_dir(ctx.artifacts_dir, case.name), "frames"
                )
                os.makedirs(artifact_frames_dir, exist_ok=True)
                for frame_path in frame_files:
                    shutil.copy2(str(frame_path), artifact_frames_dir)
                frame_paths = [
                    os.path.join(artifact_frames_dir, frame_path.name) for frame_path in frame_files
                ]

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "", "end_to_end", case.name
            )
            data: dict = {
                "returncode": result.returncode,
                "num_frames": num_frames,
                "frame_stats": frame_stats,
                "frames_dir": artifact_frames_dir or frame_dir,
                "frame_paths": frame_paths,
                "stdout": result.stdout,
                "stderr": stderr_truncated,
                "prompt": case.inputs.get("prompt") or case.inputs.get("prompt_file"),
            }
            if stderr_log:
                data["stderr_log"] = stderr_log
            return StageOutput(
                stage_name=stage.name,
                data=data,
                text=result.stdout,
                timing_s=elapsed,
                metadata={"command": cmd},
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
            array = np.array(Image.open(frame).convert("RGB"), dtype=np.float32) / 255.0
            pixels.append(array.flatten())
        combined = np.concatenate(pixels)
        return {
            "count": len(frames),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
        }


runner = SanaWmDiffusionMediaGenerationRunner()
