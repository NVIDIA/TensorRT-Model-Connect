"""Qwen Image model-owned HF diffusers reference backend."""

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

        snapshot = Path(snapshot_download(hf_id, local_files_only=True))
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


def _initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(tempfile.gettempdir(), "trtmc_qwen_image_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


class HfDiffusersReference:
    """Reference backend using Qwen Image diffusers pipelines."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "end_to_end": self._run_full_pipeline,
            "generate": self._run_full_pipeline,
            "vae_decode": self._run_full_pipeline,
            "frame_quality": self._run_full_pipeline,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Qwen Image reference stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        model_id = case.hf_id
        model_ref = _resolve_cached_model_ref(model_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 20)
        image_height = case.inputs.get("image_height", 1024)
        image_width = case.inputs.get("image_width", image_height)
        python = ctx.reference_python_path() or sys.executable
        initial_latents_raw = _initial_latents_path(case, ctx)

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)

        family = case.family
        qi_negative_prompt = case.inputs.get("negative_prompt", " ")
        qi_cfg_scale = float(
            case.inputs.get("cfg_scale", case.inputs.get("guidance_scale", 4.0)))
        qi_height = int(case.inputs.get("height", case.inputs.get("image_height", image_height)))
        qi_width = int(case.inputs.get("width", case.inputs.get("image_width", image_width)))
        qi_image_path = case.inputs.get("image") or case.inputs.get("image_path") or ""

        script = f"""
import torch
import numpy as np
from PIL import Image
import os
import transformers

transformers.logging.set_verbosity_error()

family = {family!r}
model_id = {model_id!r}
model_ref = {model_ref!r}
prompt = {prompt!r}
num_steps = {num_steps}
frames_dir = {frames_dir!r}
seed = {int(case.inputs.get("seed", case.determinism.get("seed", 42)))}
qi_negative_prompt = {qi_negative_prompt!r}
qi_cfg_scale = {qi_cfg_scale}
qi_height = {qi_height}
qi_width = {qi_width}
qi_image_path = {str(qi_image_path)!r}
qwen_image_initial_latents_raw = {initial_latents_raw!r}

if family in ("qwen_image",):
    import diffusers
    diffusers.logging.set_verbosity_error()
    pipeline_cls = None
    for cls_name in ("QwenImageEditPlusPipeline", "QwenImageEditPipeline",
                     "QwenImagePipeline"):
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        if "Edit" in cls_name and not bool(qi_image_path):
            continue
        pipeline_cls = cls
        break
    if pipeline_cls is None:
        raise RuntimeError(
            "diffusers does not expose QwenImagePipeline; upgrade diffusers")
    pipe = pipeline_cls.from_pretrained(model_ref, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    qi_input_image = Image.open(qi_image_path).convert("RGB") if qi_image_path else None
    qi_latents = None
    if os.path.exists(qwen_image_initial_latents_raw):
        vae_scale = 8
        latent_channels = 16
        h_lat = qi_height // vae_scale
        w_lat = qi_width // vae_scale
        pack_h_lat = 2 * (h_lat // 2)
        pack_w_lat = 2 * (w_lat // 2)
        raw = np.fromfile(qwen_image_initial_latents_raw, dtype=np.float32)
        expected_size = latent_channels * pack_h_lat * pack_w_lat
        if raw.size != expected_size:
            raise RuntimeError(
                f"Qwen-Image shared latents size {{raw.size}} does not "
                f"match expected [1, {{latent_channels}}, {{pack_h_lat}}, "
                f"{{pack_w_lat}}] = {{expected_size}}")
        unpacked = torch.from_numpy(raw).view(
            1, latent_channels, pack_h_lat, pack_w_lat).to(
                device="cuda", dtype=torch.bfloat16)
        qi_latents = pipeline_cls._pack_latents(
            unpacked, 1, latent_channels, pack_h_lat, pack_w_lat)
    qi_call_kwargs = dict(
        prompt=prompt,
        negative_prompt=qi_negative_prompt,
        true_cfg_scale=qi_cfg_scale,
        num_inference_steps=num_steps,
        height=qi_height,
        width=qi_width,
        latents=qi_latents,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    if qi_input_image is not None:
        qi_call_kwargs["image"] = qi_input_image
    output = pipe(**qi_call_kwargs)
    frames = output.images
else:
    raise RuntimeError(f"unsupported Qwen Image reference family {{family}}")

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
                    "Qwen Image HF reference failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[-500:],
                )

        data: dict = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )


plugin = HfDiffusersReference()
