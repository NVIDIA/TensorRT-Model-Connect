# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official Hugging Face Diffusers reference for Wan2.2 TI2V-5B."""

from __future__ import annotations

import os
import subprocess
import struct
import sys
import tempfile
import time
from pathlib import Path

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

HF_REFERENCE_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def _resolve_cached_model_ref(hf_id: str = HF_REFERENCE_ID) -> str:
    local_path = Path(hf_id)
    if local_path.exists():
        return str(local_path)

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


def _snapshot_revision(model_ref: str) -> str:
    path = Path(model_ref).resolve()
    if path.parent.name == "snapshots" and len(path.name) == 40:
        return path.name
    return "local"


def _reference_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _validate_frames(
    frames_dir: Path,
    *,
    expected_count: int,
    expected_width: int,
    expected_height: int,
) -> list[Path]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) != expected_count:
        raise RuntimeError(
            "Wan2.2 HF Diffusers reference produced "
            f"{len(frames)} frames; expected {expected_count}"
        )
    for frame in frames:
        header = frame.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"Wan2.2 HF Diffusers reference frame {frame.name} is not a PNG")
        frame_size = struct.unpack(">II", header[16:24])
        if frame_size != (expected_width, expected_height):
            raise RuntimeError(
                f"Wan2.2 HF Diffusers reference frame {frame.name} has "
                f"size {frame_size}; expected "
                f"({expected_width}, {expected_height})"
            )
    return frames


class Wan22HfDiffusersReference:
    """Generate the independent full-resolution nightly reference video."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name not in {"end_to_end", "end_to_end_video", "generate"}:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Wan2.2 reference stage: {stage.name}"},
            )

        model_ref = _resolve_cached_model_ref()
        prompt = str(case.inputs.get("prompt", ""))
        negative_prompt = str(case.inputs.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
        height = int(case.inputs.get("video_height", 704))
        width = int(case.inputs.get("video_width", 1280))
        num_frames = int(case.inputs.get("video_num_frames", 121))
        num_steps = int(case.inputs.get("num_inference_steps", 50))
        guidance_scale = float(case.inputs.get("guidance_scale", 5.0))
        flow_shift = float(case.inputs.get("flow_shift", 5.0))
        text_max_length = int(case.inputs.get("text_max_length", 512))
        seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))

        artifact_root = (
            Path(ctx.artifacts_dir) / case.name
            if ctx.artifacts_dir
            else Path(tempfile.gettempdir()) / "trtmc_wan22_reference" / case.name
        )
        frames_dir = artifact_root / "hf_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in frames_dir.glob("frame_*.png"):
            stale_frame.unlink()

        script = f"""
import os
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanPipeline

model_ref = {model_ref!r}
frames_dir = {str(frames_dir)!r}
expected_flow_shift = {flow_shift!r}

vae = AutoencoderKLWan.from_pretrained(
    model_ref,
    subfolder="vae",
    torch_dtype=torch.float32,
    local_files_only=True,
)
pipe = WanPipeline.from_pretrained(
    model_ref,
    vae=vae,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
shared_embedding_shape = tuple(pipe.text_encoder.shared.weight.shape)
encoder_embedding_shape = tuple(pipe.text_encoder.encoder.embed_tokens.weight.shape)
if shared_embedding_shape != encoder_embedding_shape:
    raise RuntimeError(
        "UMT5 shared and encoder input embedding shapes do not match: "
        f"{{shared_embedding_shape}} != {{encoder_embedding_shape}}"
    )
pipe.text_encoder.set_input_embeddings(pipe.text_encoder.shared)
if pipe.text_encoder.encoder.embed_tokens is not pipe.text_encoder.shared:
    raise RuntimeError("UMT5 encoder input embedding is not tied to shared.weight")
actual_flow_shift = float(pipe.scheduler.config.flow_shift)
if abs(actual_flow_shift - expected_flow_shift) > 1e-6:
    raise RuntimeError(
        f"Wan scheduler flow_shift {{actual_flow_shift}} does not match "
        f"requested {{expected_flow_shift}}"
    )
pipe.to("cuda")

output = pipe(
    prompt={prompt!r},
    negative_prompt={negative_prompt!r},
    height={height},
    width={width},
    num_frames={num_frames},
    guidance_scale={guidance_scale!r},
    num_inference_steps={num_steps},
    max_sequence_length={text_max_length},
    generator=torch.Generator(device="cuda").manual_seed({seed}),
)
frames = output.frames[0]
for index, frame in enumerate(frames):
    if isinstance(frame, Image.Image):
        image = frame
    else:
        array = np.asarray(frame)
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        image = Image.fromarray(array.astype(np.uint8))
    image.save(os.path.join(frames_dir, f"frame_{{index:04d}}.png"))
print(f"Generated {{len(frames)}} frames")
"""

        python = ctx.reference_python_path() or sys.executable
        timeout_s = int(case.metadata.get("runtime_timeout_s", 14400))
        if timeout_s <= 0:
            raise ValueError("runtime_timeout_s must be positive")
        started = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=_reference_env(),
        )
        elapsed = time.monotonic() - started
        stderr, stderr_log = save_full_stderr(
            result.stderr or "",
            ctx.artifacts_dir,
            "hf_diffusers_end_to_end",
            case.name,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Wan2.2 HF Diffusers reference failed (rc={result.returncode}): {stderr}"
            )

        frames = _validate_frames(
            frames_dir,
            expected_count=num_frames,
            expected_width=width,
            expected_height=height,
        )
        data: dict[str, object] = {
            "returncode": result.returncode,
            "num_frames": len(frames),
            "frames_dir": str(frames_dir),
            "frame_paths": [str(frame) for frame in frames],
            "stdout": result.stdout,
            "stderr": stderr,
            "prompt": prompt,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "model_id": HF_REFERENCE_ID,
                "model_revision": _snapshot_revision(model_ref),
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "flow_shift": flow_shift,
                "text_max_length": text_max_length,
                "seed": seed,
            },
        )


plugin = Wan22HfDiffusersReference()
