# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official-Wan reference for the full Wan2.2 TI2V case."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path


SOURCE_ENVIRONMENT = "TRTMC_REFERENCE_SOURCE_DIR"
SOURCE_ENTRYPOINT = "wan/textimage2video.py"
EXPECTED_UMT5_SPECIAL_TOKENS = {
    "pad_token": ("<pad>", 0),
    "eos_token": ("</s>", 1),
    "bos_token": ("<s>", 2),
    "unk_token": ("<unk>", 3),
}
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def _source() -> Path:
    value = os.environ.get(SOURCE_ENVIRONMENT)
    if not value:
        raise RuntimeError(f"{SOURCE_ENVIRONMENT} is required for the official Wan reference")
    source = Path(value)
    entrypoint = source / SOURCE_ENTRYPOINT
    if not entrypoint.is_file():
        raise RuntimeError(f"official Wan reference entrypoint is unavailable: {entrypoint}")
    return source


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _validate_frames(
    frames_dir: Path,
    *,
    expected_count: int,
    expected_width: int,
    expected_height: int,
) -> list[Path]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    expected_names = [f"frame_{index:04d}.png" for index in range(expected_count)]
    if [frame.name for frame in frames] != expected_names:
        raise RuntimeError(
            "official Wan reference produced an incomplete frame sequence: "
            f"found {[frame.name for frame in frames]}; expected {expected_names}"
        )
    for frame in frames:
        header = frame.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"official Wan reference frame is not a PNG: {frame}")
        if struct.unpack(">II", header[16:24]) != (expected_width, expected_height):
            raise RuntimeError(
                f"official Wan reference frame {frame.name} has the wrong dimensions"
            )
    return frames


def generate(
    model_dir: Path,
    output_root: Path,
    *,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
    flow_shift: float,
    seed: int,
    timeout_s: int,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict:
    """Run the pinned official implementation against the materialized checkpoint."""

    if num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError("Wan2.2 TI2V frame count must be 4n+1")
    if timeout_s <= 0:
        raise ValueError("runtime_timeout_s must be positive")

    source = _source()
    frames_dir = output_root / "reference-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames_dir.glob("frame_*.png"):
        frame.unlink()

    script = f"""
import os
import sys

import torch
from PIL import Image

model_ref = {str(model_dir)!r}
official_source = {str(source)!r}
frames_dir = {str(frames_dir)!r}

torch.cuda.set_device(0)

sys.path.insert(0, official_source)
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE

if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
    raise RuntimeError("Official Wan attention dependency is unavailable")

if int(ti2v_5B.text_len) != 512:
    raise RuntimeError(f"Official Wan text length {{ti2v_5B.text_len}} does not match 512")

pipeline = WanTI2V(
    config=ti2v_5B,
    checkpoint_dir=model_ref,
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_sp=False,
    t5_cpu=False,
    init_on_cpu=True,
    convert_model_dtype=False,
)

tokenizer = pipeline.text_encoder.tokenizer.tokenizer
for role, (token, token_id) in {EXPECTED_UMT5_SPECIAL_TOKENS!r}.items():
    existing = getattr(tokenizer, role, None)
    resolved_id = getattr(tokenizer, f"{{role}}_id", None)
    vocabulary_id = tokenizer.convert_tokens_to_ids(token)
    if existing != token or resolved_id != token_id or vocabulary_id != token_id:
        raise RuntimeError(
            f"Official Wan tokenizer {{role}} binding mismatch: "
            f"token={{existing!r}}, role={{resolved_id}}, vocabulary={{vocabulary_id}}, "
            f"expected=({{token!r}}, {{token_id}})"
        )

video = pipeline.generate(
    {prompt!r},
    img=None,
    size=({width}, {height}),
    max_area={width} * {height},
    frame_num={num_frames},
    shift={flow_shift!r},
    sample_solver="unipc",
    sampling_steps={num_steps},
    guide_scale={guidance_scale!r},
    n_prompt={negative_prompt!r},
    seed={seed},
    offload_model=False,
)
torch.cuda.synchronize(0)
expected_shape = (3, {num_frames}, {height}, {width})
if tuple(video.shape) != expected_shape:
    raise RuntimeError(f"Official Wan output shape {{tuple(video.shape)}} != {{expected_shape}}")
frames = (
    ((video.clamp(-1.0, 1.0) + 1.0) * 127.5)
    .to(torch.uint8)
    .permute(1, 2, 3, 0)
    .cpu()
    .numpy()
)
for index, frame in enumerate(frames):
    Image.fromarray(frame, mode="RGB").save(
        os.path.join(frames_dir, f"frame_{{index:04d}}.png")
    )
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        env=_environment(),
    )
    if result.returncode:
        raise RuntimeError(
            f"official Wan reference failed (rc={result.returncode}): {result.stderr[-4000:]}"
        )
    frames = _validate_frames(
        frames_dir,
        expected_count=num_frames,
        expected_width=width,
        expected_height=height,
    )
    return {
        "frame_paths": [str(frame) for frame in frames],
        "num_frames": len(frames),
    }
