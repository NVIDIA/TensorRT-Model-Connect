# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux model-owned HF diffusers reference backend."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..parity import ensure_initial_latents

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

        snapshot = Path(snapshot_download(
            hf_id,
            allow_patterns=["model_index.json"],
            local_files_only=True,
        ))
    except Exception:
        return hf_id
    tok_cfg = snapshot / "tokenizer" / "tokenizer_config.json"
    if not tok_cfg.exists():
        return str(snapshot)
    try:
        cfg = json.loads(tok_cfg.read_text())
    except Exception:
        return str(snapshot)
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return str(snapshot)
    patched_root = Path(tempfile.gettempdir()) / "trtmc_hf_patched" / hashlib.sha256(
        str(snapshot).encode("utf-8")).hexdigest()
    patched_cfg = patched_root / "tokenizer" / "tokenizer_config.json"
    if not patched_cfg.exists():
        shutil.copytree(snapshot, patched_root, dirs_exist_ok=True)
        patched = json.loads(patched_cfg.read_text())
        patched.pop("extra_special_tokens", None)
        patched_cfg.write_text(json.dumps(patched, indent=2))
    return str(patched_root)


class HfDiffusersReference:
    """Reference backend using Flux diffusers pipelines."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        handler = self._run_full_pipeline if stage.name in {
            "end_to_end", "generate", "vae_decode", "frame_quality"
        } else None
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Flux reference stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_ref = _resolve_cached_model_ref(case.hf_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        batch_prompts = case.inputs.get("batch_prompts")
        if not isinstance(batch_prompts, list) or len(batch_prompts) < 2:
            batch_prompts = [prompt]
        batch_seeds = case.inputs.get("batch_seeds")
        if not isinstance(batch_seeds, list) or len(batch_seeds) != len(batch_prompts):
            seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
            batch_seeds = [seed] * len(batch_prompts)
        num_steps = case.inputs.get("num_inference_steps", 30)
        image_height = case.inputs.get("image_height", 1024)
        image_width = case.inputs.get("image_width", image_height)
        model_type = str(case.metadata.get("model_type", "")).lower()
        default_reference_precision = (
            "bf16" if model_type in {"flux.2", "flux2"} else "fp32"
        )
        task_config = case.metadata.get("task_eval", {})
        task_reference_precision = (
            task_config.get("reference_precision")
            if isinstance(task_config, dict)
            else None
        )
        reference_precision = str(
            case.metadata.get(
                "reference_precision",
                task_reference_precision or default_reference_precision,
            )
        ).lower()
        reference_torch_dtype = {
            "fp16": "torch.float16",
            "bf16": "torch.bfloat16",
            "fp32": "torch.float32",
        }.get(reference_precision, "torch.float32")
        guidance_scale = case.inputs.get("guidance_scale")
        python = ctx.reference_python_path() or sys.executable
        initial_latents = ensure_initial_latents(case, ctx)

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

transformers.logging.set_verbosity_error()

model_ref = {model_ref!r}
prompts = {batch_prompts!r}
batch_seeds = {batch_seeds!r}
num_steps = {num_steps}
image_height = {image_height}
image_width = {image_width}
model_type = {model_type!r}
guidance_scale = {guidance_scale!r}
frames_dir = {frames_dir!r}
reference_torch_dtype = {reference_torch_dtype}
initial_latents_path = {str(initial_latents.path)!r}
initial_latents_shape = {initial_latents.shape!r}

if model_type in ("flux.2", "flux2"):
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(
        model_ref, torch_dtype=reference_torch_dtype, low_cpu_mem_usage=True)
else:
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        model_ref, torch_dtype=reference_torch_dtype)
pipe.enable_sequential_cpu_offload()
raw_latents = np.fromfile(initial_latents_path, dtype=np.float32)
expected_size = int(np.prod(initial_latents_shape))
if raw_latents.size != expected_size:
    raise RuntimeError(
        f"Flux shared latents size {{raw_latents.size}} does not match "
        f"expected {{initial_latents_shape}} = {{expected_size}}"
    )
unpacked_latents = torch.from_numpy(raw_latents.reshape(initial_latents_shape)).to("cuda")
if model_type in ("flux.2", "flux2"):
    hf_latents = unpacked_latents.to(dtype=reference_torch_dtype)
else:
    hf_latents = pipe._pack_latents(
        unpacked_latents.to(dtype=reference_torch_dtype),
        initial_latents_shape[0],
        initial_latents_shape[1],
        initial_latents_shape[2],
        initial_latents_shape[3],
    )
kwargs = dict(
    prompt=prompts if len(prompts) > 1 else prompts[0],
    num_inference_steps=num_steps,
    height=image_height,
    width=image_width,
)
if len(prompts) > 1:
    kwargs["generator"] = [
        torch.Generator("cuda").manual_seed(seed) for seed in batch_seeds
    ]
else:
    kwargs["latents"] = hf_latents
if model_type in ("flux.2", "flux2"):
    kwargs["guidance_scale"] = 3.5 if guidance_scale is None else guidance_scale
output = pipe(**kwargs)
frames = output.images

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
            logger.error("Flux HF reference failed (rc=%d): %s", result.returncode, result.stderr[-500:])
        data = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "prompts": batch_prompts,
        }
        if len(batch_prompts) == 1:
            data.update(
                {
                    "initial_latents_path": str(initial_latents.path),
                    "initial_latents_sha256": initial_latents.sha256,
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
