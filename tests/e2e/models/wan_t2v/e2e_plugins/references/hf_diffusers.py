# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan T2V model-owned HF diffusers reference backend."""

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
    """Reference backend using the Wan diffusers pipeline."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name in {"end_to_end", "end_to_end_video", "generate", "vae_decode", "frame_quality"}:
            return self._run_full_pipeline(case, stage, ctx)
        if stage.name == "t5_encode":
            return self._run_t5_encode(case, stage, ctx)
        return StageOutput(
            stage_name=stage.name,
            data={"error": f"Unsupported Wan reference stage: {stage.name}"},
        )

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_ref = _resolve_cached_model_ref(case.hf_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        python = ctx.reference_python_path() or sys.executable
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, "hf_t5_output.npy")
        script = f"""
import torch, numpy as np
import transformers
from diffusers import WanPipeline

transformers.logging.set_verbosity_error()
try:
    pipe = WanPipeline.from_pretrained(
        {model_ref!r}, torch_dtype=torch.float32, low_cpu_mem_usage=False)
except ValueError as e:
    if "keep_in_fp32_modules" not in str(e):
        raise
    pipe = WanPipeline.from_pretrained(
        {model_ref!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
if hasattr(pipe.text_encoder, "tie_weights"):
    pipe.text_encoder.tie_weights()
shared = getattr(pipe.text_encoder, "shared", None)
embed = getattr(getattr(pipe.text_encoder, "encoder", None), "embed_tokens", None)
if shared is not None and embed is not None and shared.weight.shape == embed.weight.shape:
    if shared.weight.data_ptr() != embed.weight.data_ptr():
        pipe.text_encoder.encoder.embed_tokens = pipe.text_encoder.shared
tokens = pipe.tokenizer(
    {prompt!r}, return_tensors="pt", padding="max_length",
    max_length={int(case.inputs.get("text_max_length", 512))}, truncation=True)
with torch.no_grad():
    t5_out = pipe.text_encoder(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
    )[0]
np.save({output_path!r}, t5_out.numpy())
print(f"shape={{list(t5_out.shape)}}")
print(f"mean={{float(t5_out.mean()):.6f}}")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env=_ref_subprocess_env(),
        )
        elapsed = time.monotonic() - t0
        data: dict = {"returncode": result.returncode, "stdout": result.stdout}
        if os.path.exists(output_path):
            data["output_path"] = output_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_ref = _resolve_cached_model_ref(case.hf_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        video_height = case.inputs.get("video_height", 480)
        video_width = case.inputs.get("video_width", 832)
        video_num_frames = case.inputs.get("video_num_frames", 17)
        guidance_scale = float(case.inputs.get("guidance_scale", 5.0))
        python = ctx.reference_python_path() or sys.executable
        initial_latents = (
            ensure_initial_latents(case, ctx)
            if uses_shared_initial_latents(case)
            else None
        )

        if initial_latents is not None:
            latent_setup = f"""
raw_latents = np.fromfile({str(initial_latents.path)!r}, dtype=np.float32)
expected_shape = {initial_latents.shape!r}
expected_size = int(np.prod(expected_shape))
if raw_latents.size != expected_size:
    raise RuntimeError(
        f"Wan shared latents size {{raw_latents.size}} does not match "
        f"expected {{expected_shape}} = {{expected_size}}"
    )
initial_latents = torch.from_numpy(raw_latents.reshape(expected_shape)).to(
    device="cuda", dtype=torch.float32)
"""
            generation_input = "latents=initial_latents,"
        else:
            latent_setup = ""
            generation_input = ""

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
import ftfy  # noqa: F401
from diffusers import WanPipeline

transformers.logging.set_verbosity_error()
try:
    pipe = WanPipeline.from_pretrained(
        {model_ref!r}, torch_dtype=torch.float32, low_cpu_mem_usage=False)
except ValueError as e:
    if "keep_in_fp32_modules" not in str(e):
        raise
    pipe = WanPipeline.from_pretrained(
        {model_ref!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
if hasattr(pipe.text_encoder, "tie_weights"):
    pipe.text_encoder.tie_weights()
shared = getattr(pipe.text_encoder, "shared", None)
embed = getattr(getattr(pipe.text_encoder, "encoder", None), "embed_tokens", None)
if shared is not None and embed is not None and shared.weight.shape == embed.weight.shape:
    if shared.weight.data_ptr() != embed.weight.data_ptr():
        pipe.text_encoder.encoder.embed_tokens = pipe.text_encoder.shared
pipe.to("cuda")
{latent_setup}
output = pipe(
    prompt={prompt!r},
    num_inference_steps={num_steps},
    height={video_height},
    width={video_width},
    num_frames={video_num_frames},
    guidance_scale={guidance_scale},
    {generation_input}
    generator=torch.Generator("cuda").manual_seed({int(case.inputs.get("seed", case.determinism.get("seed", 42)))}),
)
frames = output.frames[0]
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
            logger.error("Wan HF reference failed (rc=%d): %s", result.returncode, result.stderr[-500:])
        data = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if initial_latents is not None:
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
