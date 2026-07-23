# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official-Wan reference for Wan2.2 TI2V-5B Nightly qualification."""

from __future__ import annotations

import os
import subprocess
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

HF_REFERENCE_ID = "Wan-AI/Wan2.2-TI2V-5B"
HF_REFERENCE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
OFFICIAL_REPOSITORY = "https://github.com/Wan-Video/Wan2.2.git"
OFFICIAL_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
OFFICIAL_RELATIVE_PATH = "wan2_2_ti2v/reference/Wan2.2-42bf4cfaa384"
OFFICIAL_ENTRYPOINT = "wan/textimage2video.py"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)

_NATIVE_ACCEPTANCE_VALUES = {
    "kind": "native_visual_semantic_acceptance",
    "reference_role": "diagnostic",
    "requires_nightly_vlm": True,
    "vlm_frame_samples": 6,
}


def _native_acceptance_policy(metadata: dict[str, Any]) -> dict[str, Any] | None:
    policy = metadata.get("native_acceptance")
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError("Wan2.2 native_acceptance must be an object")
    invalid = {
        key: (policy.get(key), expected)
        for key, expected in _NATIVE_ACCEPTANCE_VALUES.items()
        if policy.get(key) != expected
    }
    if not isinstance(policy.get("rationale"), str) or not policy["rationale"].strip():
        invalid["rationale"] = (policy.get("rationale"), "non-empty string")
    if invalid:
        raise ValueError(f"Wan2.2 native_acceptance policy is invalid: {invalid}")
    return dict(policy)


def _resolve_cached_model_ref() -> str:
    from huggingface_hub import snapshot_download
    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    return str(
        Path(
            snapshot_download(
                HF_REFERENCE_ID,
                revision=HF_REFERENCE_REVISION,
                allow_patterns=hf_snapshot_allow_patterns(),
                local_files_only=True,
            )
        )
    )


def _snapshot_revision(model_ref: str) -> str:
    path = Path(model_ref).resolve()
    if path.parent.name == "snapshots" and len(path.name) == 40:
        return path.name
    return ""


def _resolve_official_source(storage_root: str | None = None) -> Path:
    root_value = storage_root if storage_root is not None else os.environ.get("TRTMC_STORAGE_ROOT")
    if not root_value:
        raise RuntimeError("TRTMC_STORAGE_ROOT is required for the pinned official Wan reference")
    source = Path(root_value) / OFFICIAL_RELATIVE_PATH
    entrypoint = source / OFFICIAL_ENTRYPOINT
    if source.is_symlink() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise RuntimeError(
            "Pinned official Wan reference is unavailable at "
            f"{entrypoint}; expected {OFFICIAL_REPOSITORY}@{OFFICIAL_REVISION}"
        )
    return source


def _reference_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _validate_frames(
    frames_dir: Path,
    *,
    expected_count: int,
    expected_width: int,
    expected_height: int,
) -> list[Path]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    expected_names = [f"frame_{index:04d}.png" for index in range(expected_count)]
    if len(frames) != expected_count:
        raise RuntimeError(
            "Wan2.2 official Wan reference produced "
            f"{len(frames)} frames; expected {expected_count}"
        )
    if [frame.name for frame in frames] != expected_names:
        raise RuntimeError(
            "Wan2.2 official Wan reference produced a non-contiguous frame sequence: "
            f"found {[frame.name for frame in frames]}; expected {expected_names}"
        )
    for frame in frames:
        header = frame.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"Wan2.2 official Wan reference frame {frame.name} is not a PNG")
        frame_size = struct.unpack(">II", header[16:24])
        if frame_size != (expected_width, expected_height):
            raise RuntimeError(
                f"Wan2.2 official Wan reference frame {frame.name} has "
                f"size {frame_size}; expected "
                f"({expected_width}, {expected_height})"
            )
    return frames


class Wan22OfficialWanReference:
    """Generate the full-resolution Nightly oracle with pinned official Wan code."""

    @property
    def backend_name(self) -> str:
        return "wan_official"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name not in {"end_to_end", "end_to_end_video", "generate"}:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported Wan2.2 reference stage: {stage.name}"},
            )

        native_acceptance = _native_acceptance_policy(case.metadata)
        model_ref = _resolve_cached_model_ref()
        model_revision = _snapshot_revision(model_ref)
        if model_revision != HF_REFERENCE_REVISION:
            raise RuntimeError(
                "Wan2.2 checkpoint revision mismatch: expected "
                f"{HF_REFERENCE_REVISION}, found {model_revision}"
            )
        official_source = _resolve_official_source()
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
        if text_max_length != 512:
            raise ValueError(
                "The pinned official Wan2.2 TI2V-5B reference requires text_max_length=512"
            )
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("Wan2.2 TI2V frame count must be 4n+1")

        artifact_root = (
            Path(ctx.artifacts_dir) / case.name
            if ctx.artifacts_dir
            else Path(tempfile.gettempdir()) / "trtmc_wan22_reference" / case.name
        )
        # Retain this stable directory name for the HTML/report frame-pair tooling.
        frames_dir = artifact_root / "hf_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in frames_dir.glob("frame_*.png"):
            stale_frame.unlink()

        script = f"""
import importlib.util
import os
import sys
import types

import torch
from PIL import Image

model_ref = {model_ref!r}
official_source = {str(official_source)!r}
frames_dir = {str(frames_dir)!r}

torch.cuda.set_device(0)

# The official repository imports optional task and video-writer dependencies
# eagerly. These narrow compatibility shims keep the pristine pinned TI2V code
# while ensuring that unused S2V/Animate/imageio surfaces cannot run.
easydict = types.ModuleType("easydict")
easydict.__spec__ = importlib.util.spec_from_loader("easydict", loader=None)

class EasyDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value

easydict.EasyDict = EasyDict
sys.modules["easydict"] = easydict

imageio = types.ModuleType("imageio")
imageio.__spec__ = importlib.util.spec_from_loader("imageio", loader=None, is_package=True)
imageio.__path__ = []

def unexpected_video_writer(*args, **kwargs):
    raise RuntimeError("Wan2.2 qualification must use the PNG frame writer")

imageio.get_writer = unexpected_video_writer
sys.modules["imageio"] = imageio

wan = types.ModuleType("wan")
wan.__path__ = [os.path.join(official_source, "wan")]
wan.__package__ = "wan"
wan.__spec__ = importlib.util.spec_from_loader("wan", loader=None, is_package=True)
sys.modules["wan"] = wan

wan_configs = types.ModuleType("wan.configs")
wan_configs.__path__ = [os.path.join(official_source, "wan", "configs")]
wan_configs.__package__ = "wan.configs"
wan_configs.__spec__ = importlib.util.spec_from_loader(
    "wan.configs", loader=None, is_package=True
)
sys.modules["wan.configs"] = wan_configs

from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V
import wan.modules.model as wan_model_module
from wan.modules.attention import attention as wan_attention

# The pinned source calls flash-attention directly, but the hermetic GB300 CI
# image intentionally has no flash-attn wheel. Bind the official module to its
# own PyTorch SDPA fallback, matching official-gb300-sdpa.patch without changing
# the archived source tree.
wan_model_module.flash_attention = wan_attention

if int(ti2v_5B.text_len) != {text_max_length}:
    raise RuntimeError(
        f"Official Wan text length {{ti2v_5B.text_len}} does not match requested {text_max_length}"
    )

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
print(f"Generated {{len(frames)}} frames with pinned official Wan")
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
            "official_wan_end_to_end",
            case.name,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Wan2.2 official Wan reference failed (rc={result.returncode}): {stderr}"
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
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        if native_acceptance is not None:
            data["native_acceptance"] = native_acceptance
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={
                "model_id": HF_REFERENCE_ID,
                "model_revision": model_revision,
                "expected_model_revision": HF_REFERENCE_REVISION,
                "official_repository": OFFICIAL_REPOSITORY,
                "official_revision": OFFICIAL_REVISION,
            },
        )


plugin = Wan22OfficialWanReference()
