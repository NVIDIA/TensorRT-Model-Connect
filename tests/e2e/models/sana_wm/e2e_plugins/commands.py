# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command builders for the SANA-WM model-card contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import E2ECase, RunContext

_MODEL_DIR = Path(__file__).resolve().parents[1]


def _csv_arg(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _prompt_file(case: E2ECase) -> str:
    return str(
        case.inputs.get("prompt_file")
        or "tests/e2e/models/sana_wm/assets/demo_0.txt"
    )


def _prompt_text(case: E2ECase) -> str:
    prompt = case.inputs.get("prompt")
    if prompt:
        return str(prompt)
    path = Path(_prompt_file(case))
    if not path.is_absolute() and not path.is_file():
        path = _MODEL_DIR / path
    return path.read_text(encoding="utf-8").strip()


def _image(case: E2ECase) -> str:
    return str(
        case.inputs.get("image")
        or case.inputs.get("test_image")
        or case.inputs.get("image_path")
        or "tests/e2e/models/sana_wm/assets/demo_0.png"
    )


def build_sana_wm_trt_command(
    case: E2ECase,
    ctx: RunContext,
    bundle_path: str,
    output_dir: str,
) -> list[str]:
    """Build the C++ runtime command for the SANA-WM model-card demo."""
    parts = [
        ctx.binary_path,
        "generate-video",
        bundle_path,
        "--prompt",
        _prompt_text(case),
        "--output",
        output_dir,
    ]

    def add_setting(name: str, value: Any) -> None:
        parts.extend(["--set", f"sana_wm.{name}={value}"])

    add_setting("image_path", _image(case))
    action = case.inputs.get("action", "w-80,jw-40,w-40,lw-60,w-100")
    add_setting("action", action)

    intrinsics = case.inputs.get("camera_intrinsics")
    if intrinsics is None:
        intrinsics = case.inputs.get("intrinsics")
    if intrinsics is not None:
        if isinstance(intrinsics, (list, tuple)):
            add_setting("intrinsics", _csv_arg(intrinsics))

    for input_key, flag in (
        ("translation_speed", "translation_speed"),
        ("rotation_speed_deg", "rotation_speed_deg"),
    ):
        if input_key in case.inputs:
            add_setting(flag, case.inputs[input_key])

    num_frames = case.inputs.get("video_num_frames", case.inputs.get("num_frames"))
    if num_frames is not None:
        add_setting("num_frames", num_frames)

    for input_key, flag in (
        ("fps", "fps"),
        ("flow_shift", "flow_shift"),
    ):
        if input_key in case.inputs:
            add_setting(flag, case.inputs[input_key])

    num_steps = (
        case.inputs.get("num_inference_steps")
        or case.inputs.get("num_steps")
        or case.inputs.get("step")
    )
    if num_steps is not None:
        parts.extend(["--num-steps", str(num_steps)])

    cfg_scale = case.inputs.get("cfg_scale")
    if cfg_scale is None:
        cfg_scale = case.inputs.get("guidance_scale")
    if cfg_scale is not None:
        parts.extend(["--guidance-scale", str(cfg_scale)])

    if "seed" in case.inputs:
        parts.extend(["--seed", str(case.inputs["seed"])])
    if case.inputs.get("no_refiner"):
        add_setting("no_refiner", "true")
    return parts


def build_sana_wm_reference_command(
    case: E2ECase,
    python: str,
    output_dir: str,
) -> list[str]:
    """Build the Python reference command matching the HF model-card args."""
    parts = [
        python,
        "tests/e2e/models/sana_wm/reference/inference_sana_wm.py",
        "--image",
        _image(case),
        "--prompt",
        _prompt_file(case),
        "--output_dir",
        output_dir,
    ]

    camera = case.inputs.get("camera") or case.inputs.get("camera_path")
    if camera:
        parts.extend(["--camera", str(camera)])
    else:
        action = case.inputs.get("action", "w-80,jw-40,w-40,lw-60,w-100")
        parts.extend(["--action", str(action)])

    intrinsics_file = case.inputs.get("camera_intrinsics_file")
    if intrinsics_file:
        parts.extend(["--intrinsics", str(intrinsics_file)])

    parts.extend(
        [
            "--translation_speed",
            str(case.inputs.get("translation_speed", 0.055)),
            "--rotation_speed_deg",
            str(case.inputs.get("rotation_speed_deg", 1.2)),
            "--num_frames",
            str(case.inputs.get("video_num_frames", case.inputs.get("num_frames", 321))),
        ]
    )

    num_steps = (
        case.inputs.get("num_inference_steps")
        or case.inputs.get("num_steps")
        or case.inputs.get("step")
    )
    if num_steps is not None:
        parts.extend(["--step", str(num_steps)])

    cfg_scale = case.inputs.get("cfg_scale")
    if cfg_scale is None:
        cfg_scale = case.inputs.get("guidance_scale")
    if cfg_scale is not None:
        parts.extend(["--cfg_scale", str(cfg_scale)])

    fps = case.inputs.get("fps")
    if fps is not None:
        parts.extend(["--fps", str(fps)])

    flow_shift = case.inputs.get("flow_shift")
    if flow_shift is not None:
        parts.extend(["--flow_shift", str(flow_shift)])

    if case.inputs.get("no_action_overlay"):
        parts.append("--no_action_overlay")
    if case.inputs.get("no_refiner"):
        parts.append("--no_refiner")
    return parts
