# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video model-owned HF diffusers reference backend."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..parity import ensure_initial_latents, uses_shared_initial_latents

logger = logging.getLogger(__name__)


def _ref_subprocess_env() -> dict:
    env = os.environ.copy()
    sys_cublas = "/usr/local/cuda/lib64/libcublas.so.13"
    sys_cublaslt = "/usr/local/cuda/lib64/libcublasLt.so.13"
    if os.path.exists(sys_cublas) and os.path.exists(sys_cublaslt):
        existing = env.get("LD_PRELOAD", "")
        preload = f"{sys_cublas}:{sys_cublaslt}"
        env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _resolve_cached_model_ref(hf_id: str) -> str:
    if not hf_id:
        return hf_id
    local_path = Path(hf_id)
    if local_path.exists():
        return hf_id
    try:
        from huggingface_hub import snapshot_download
        from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

        return str(
            Path(
                snapshot_download(
                    hf_id,
                    allow_patterns=hf_snapshot_allow_patterns(),
                    local_files_only=True,
                )
            )
        )
    except Exception:
        return hf_id


def _initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(tempfile.gettempdir(), "trtmc_ltx_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


class HfDiffusersReference:
    """Reference backend using the LTX diffusers pipeline."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "end_to_end": self._run_full_pipeline,
            "end_to_end_video": self._run_full_pipeline,
            "generate": self._run_full_pipeline,
            "vae_decode": self._run_full_pipeline,
            "frame_quality": self._run_full_pipeline,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported LTX reference stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_id = case.hf_id
        model_ref = _resolve_cached_model_ref(model_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        video_height = case.inputs.get("video_height", 480)
        video_width = case.inputs.get("video_width", 832)
        video_num_frames = case.inputs.get("video_num_frames", 17)
        guidance_scale = float(case.inputs.get("guidance_scale", 3.0))
        python = ctx.reference_python_path() or sys.executable
        shared_initial_latents = (
            ensure_initial_latents(case, ctx)
            if uses_shared_initial_latents(case)
            else None
        )
        initial_latents_raw = (
            str(shared_initial_latents.path)
            if shared_initial_latents is not None
            else _initial_latents_path(case, ctx)
        )

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)

        script = f"""
import torch
import numpy as np
from PIL import Image
import os
import transformers
from diffusers import LTXPipeline

transformers.logging.set_verbosity_error()

model_ref = {model_ref!r}
prompt = {prompt!r}
num_steps = {num_steps}
video_height = {video_height}
video_width = {video_width}
video_num_frames = {video_num_frames}
guidance_scale = {guidance_scale}
initial_latents_raw = {initial_latents_raw!r}
frames_dir = {frames_dir!r}
seed = {int(case.inputs.get("seed", case.determinism.get("seed", 42)))}

pipe = LTXPipeline.from_pretrained(model_ref, torch_dtype=torch.float32)
pipe.to("cuda")
initial_latents = None
if os.path.exists(initial_latents_raw):
    packed = np.fromfile(initial_latents_raw, dtype=np.float32)
    channels = int(pipe.transformer.config.in_channels)
    if packed.size % channels != 0:
        raise RuntimeError(
            f"invalid LTX initial latent size {{packed.size}} for {{channels}} channels")
    initial_latents = torch.from_numpy(
        packed.reshape(1, packed.size // channels, channels)).to(
            device="cuda", dtype=torch.float32)
output = pipe(
    prompt=prompt,
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps=num_steps,
    height=video_height,
    width=video_width,
    num_frames=video_num_frames,
    guidance_scale=guidance_scale,
    latents=initial_latents,
    generator=torch.Generator("cuda").manual_seed(seed),
)
frames = output.frames[0]

for i, frame in enumerate(frames):
    if isinstance(frame, Image.Image):
        frame.save(os.path.join(frames_dir, f"frame_{{i:04d}}.png"))
    else:
        img = Image.fromarray(np.uint8(frame * 255) if frame.max() <= 1.0 else np.uint8(frame))
        img.save(os.path.join(frames_dir, f"frame_{{i:04d}}.png"))

print(f"Generated {{len(frames)}} frames")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=3600,
            env=_ref_subprocess_env(),
        )
        elapsed = time.monotonic() - t0

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        if result.stderr:
            stderr_path = os.path.join(model_dir, "hf_diffusion_full_pipeline_stderr.log")
            try:
                with open(stderr_path, "w", encoding="utf-8") as f:
                    f.write(result.stderr)
            except OSError:
                pass
            if result.returncode != 0:
                logger.error(
                    "LTX HF reference failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[-500:],
                )

        data = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if shared_initial_latents is not None:
            data.update(
                {
                    "initial_latents_path": str(shared_initial_latents.path),
                    "initial_latents_sha256": shared_initial_latents.sha256,
                }
            )
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )


plugin = HfDiffusersReference()
