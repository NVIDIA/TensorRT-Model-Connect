# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video model-owned diffusion media runner."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..parity import ensure_initial_latents, uses_shared_initial_latents

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[6]
TOOLS_DIR = PROJECT_DIR / "tools"


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
        bundle_name = f"{case.name}.trtfb"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


def _initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(tempfile.gettempdir(), "trtmc_ltx_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


def _ensure_initial_latents(case: E2ECase, ctx: RunContext, bundle_path: str) -> str:
    output_path = _initial_latents_path(case, ctx)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    script = textwrap.dedent(f"""\
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(TOOLS_DIR)!r})
        import torch
        from diffusion_helpers import load_bundle_config

        cfg = load_bundle_config({bundle_path!r})
        video_num_frames = int({int(case.inputs.get("video_num_frames", 0))} or cfg.get("video_num_frames", 9))
        video_height = int({int(case.inputs.get("video_height", 0))} or cfg.get("video_height", 256))
        video_width = int({int(case.inputs.get("video_width", 0))} or cfg.get("video_width", 256))
        z_dim = int(cfg.get("z_dim", 128))
        scale_t = max(int(cfg.get("scale_factor_temporal", 8)), 1)
        scale_s = max(int(cfg.get("scale_factor_spatial", 32)), 1)
        pt, ph, pw = [int(v) for v in cfg.get("patch_size", [1, 1, 1])]
        t_lat = (video_num_frames - 1) // scale_t + 1
        h_lat = video_height // scale_s
        w_lat = video_width // scale_s
        if t_lat % pt or h_lat % ph or w_lat % pw:
            raise RuntimeError(
                f"latent shape {{t_lat}}x{{h_lat}}x{{w_lat}} is not divisible by patch {{pt}}x{{ph}}x{{pw}}")

        generator = torch.Generator("cuda").manual_seed({seed})
        latents = torch.randn(
            (1, z_dim, t_lat, h_lat, w_lat),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        latents = latents.reshape(
            1, -1, t_lat // pt, pt, h_lat // ph, ph, w_lat // pw, pw
        ).permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3).contiguous()

        out = Path({output_path!r})
        out.parent.mkdir(parents=True, exist_ok=True)
        latents.cpu().numpy().astype("<f4", copy=False).tofile(out)
        print(str(out))
    """)

    python = ctx.runtime_python_path() or sys.executable
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to create LTX initial latents: "
            + (result.stderr or result.stdout or "").strip()
        )
    return output_path


class DiffusionMediaRunner:
    """TRT runner for LTX Video generation."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "end_to_end": self._run_end_to_end,
            "end_to_end_video": self._run_end_to_end,
            "generate": self._run_end_to_end,
            "frame_quality": self._run_end_to_end,
            "vae_decode": self._run_end_to_end,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported LTX diffusion stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return handler(case, stage, ctx)

    def _run_end_to_end(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = _resolve_bundle_path(case, ctx)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)

        shared_initial_latents = None
        try:
            if uses_shared_initial_latents(case):
                shared_initial_latents = ensure_initial_latents(case, ctx)
                initial_latents_raw = str(shared_initial_latents.path)
            else:
                initial_latents_raw = _ensure_initial_latents(case, ctx, bundle_path)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"returncode": 1, "stderr": str(exc), "num_frames": 0},
                text="",
                timing_s=0.0,
                metadata={"command": "create_ltx_initial_latents"},
            )

        with tempfile.TemporaryDirectory(prefix="trtmc_ltx_frames_") as frame_dir:
            cmd = [
                ctx.binary_path,
                "generate-video",
                bundle_path,
                "--prompt",
                prompt,
                "--num-steps",
                str(num_steps),
                "--initial-latents-raw",
                initial_latents_raw,
            ]

            guidance_scale = case.inputs.get("guidance_scale")
            if guidance_scale is not None:
                cmd.extend(["--guidance-scale", str(guidance_scale)])
            if "seed" in case.inputs:
                cmd.extend(["--seed", str(case.inputs["seed"])])

            runtime_cli_python = ctx.runtime_cli_hf_python()
            if runtime_cli_python:
                cmd.extend(["--hf-python", runtime_cli_python])

            cmd.extend(["--output", frame_dir])
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
            }
            if shared_initial_latents is not None:
                data.update(
                    {
                        "initial_latents_path": str(shared_initial_latents.path),
                        "initial_latents_sha256": shared_initial_latents.sha256,
                    }
                )
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
