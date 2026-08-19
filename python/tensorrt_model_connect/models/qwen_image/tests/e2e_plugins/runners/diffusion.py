# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image model-owned diffusion media runner."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..parity import ensure_initial_latents

logger = logging.getLogger(__name__)


def _find_trt_lib_dir() -> str:
    try:
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def _build_ld_library_path(ctx: RunContext) -> str:
    if ctx.ld_library_path:
        return ctx.ld_library_path
    parts = []
    trt_lib = _find_trt_lib_dir()
    if trt_lib:
        parts.append(trt_lib)
    parts.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.bundle"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


class DiffusionMediaRunner:
    """TRT runner for Qwen Image generation."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "end_to_end": self._run_end_to_end,
            "generate": self._run_end_to_end,
            "frame_quality": self._run_end_to_end,
            "vae_decode": self._run_end_to_end,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Qwen Image diffusion stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return handler(case, stage, ctx)

    def _run_end_to_end(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 20)

        try:
            initial_latents = ensure_initial_latents(case, ctx)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"returncode": 1, "stderr": str(exc), "num_frames": 0},
                text="",
                timing_s=0.0,
                metadata={"command": "create_qwen_image_initial_latents"},
            )

        with tempfile.TemporaryDirectory(prefix="trtmc_qwen_image_") as frame_dir:
            output_png = os.path.join(frame_dir, "frame_0000.png")
            cmd = [
                binary,
                "run",
                bundle_path,
                "--prompt",
                prompt,
                "--num-inference-steps",
                str(num_steps),
                "--initial-latents-raw",
                str(initial_latents.path),
            ]
            if ctx.model_plugin_dir:
                cmd.extend(["--model-plugin-dir", ctx.model_plugin_dir])

            negative_prompt = case.inputs.get("negative_prompt")
            if negative_prompt is not None:
                cmd.extend(["--negative-prompt", str(negative_prompt)])

            cfg_scale = case.inputs.get("cfg_scale")
            if cfg_scale is None:
                cfg_scale = case.inputs.get("guidance_scale")
            if cfg_scale is not None:
                cmd.extend(["--cfg-scale", str(cfg_scale)])

            height = case.inputs.get("height") or case.inputs.get("image_height")
            if height is not None:
                cmd.extend(["--height", str(height)])

            width = case.inputs.get("width") or case.inputs.get("image_width")
            if width is not None:
                cmd.extend(["--width", str(width)])

            if "seed" in case.inputs:
                cmd.extend(["--seed", str(case.inputs["seed"])])

            image_path = case.inputs.get("image") or case.inputs.get("image_path")
            if image_path:
                cmd.extend(["--image", str(image_path)])

            runtime_cli_python = ctx.runtime_cli_hf_python()
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])

            cmd.extend(["--output", output_png])
            env = {**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)}

            t0 = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,
                env=env,
            )
            elapsed = time.monotonic() - t0

            frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
            frame_paths = [str(f) for f in frame_files]
            frame_stats = self._compute_frame_stats(frame_dir) if frame_files else {}

            artifact_frames_dir = None
            if ctx.artifacts_dir and frame_files:
                artifact_frames_dir = os.path.join(
                    _case_artifact_dir(ctx.artifacts_dir, case.name), "frames")
                os.makedirs(artifact_frames_dir, exist_ok=True)
                for fp in frame_files:
                    shutil.copy2(str(fp), artifact_frames_dir)
                frame_paths = [os.path.join(artifact_frames_dir, fp.name) for fp in frame_files]

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "", "end_to_end", case.name)
            data: dict = {
                "returncode": result.returncode,
                "num_frames": len(frame_files),
                "frame_stats": frame_stats,
                "frames_dir": artifact_frames_dir or frame_dir,
                "frame_paths": frame_paths,
                "stdout": result.stdout,
                "stderr": stderr_truncated,
                "prompt": prompt,
                "initial_latents_path": str(initial_latents.path),
                "initial_latents_sha256": initial_latents.sha256,
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

        all_pixels = []
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            all_pixels.append((np.array(img, dtype=np.float32) / 255.0).flatten())
        combined = np.concatenate(all_pixels)
        return {
            "count": len(frames),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
        }


plugin = DiffusionMediaRunner()
