# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image model-owned HF diffusers reference backend."""

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
    if not hf_id or Path(hf_id).exists():
        return hf_id
    try:
        from huggingface_hub import snapshot_download

        return str(Path(snapshot_download(hf_id, local_files_only=True)))
    except Exception:
        return hf_id


class HfDiffusersReference:
    """Reference backend using the Z-Image diffusers pipeline."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name not in {"end_to_end", "generate", "vae_decode", "frame_quality"}:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Z-Image reference stage: {stage.name}"},
            )
        return self._run_full_pipeline(case, stage, ctx)

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_ref = _resolve_cached_model_ref(case.hf_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        image_height = case.inputs.get("image_height", 1024)
        image_width = case.inputs.get("image_width", image_height)
        guidance_scale = float(case.inputs.get("guidance_scale", 0.0))
        python = ctx.reference_python_path() or sys.executable

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
from diffusers import DiffusionPipeline

transformers.logging.set_verbosity_error()

pipe = DiffusionPipeline.from_pretrained(
    {model_ref!r}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
pipe.to("cuda")
output = pipe(
    prompt={prompt!r},
    num_inference_steps={num_steps},
    height={image_height},
    width={image_width},
    guidance_scale={guidance_scale},
    generator=torch.Generator("cuda").manual_seed({int(case.inputs.get("seed", case.determinism.get("seed", 42)))}),
)
frames = output.images
frames_dir = {frames_dir!r}
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
        if result.returncode != 0:
            logger.error("Z-Image HF reference failed (rc=%d): %s", result.returncode, result.stderr[-500:])
        return StageOutput(
            stage_name=stage.name,
            data={
                "returncode": result.returncode,
                "num_frames": len(frame_files),
                "frames_dir": frames_dir,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )


plugin = HfDiffusersReference()
