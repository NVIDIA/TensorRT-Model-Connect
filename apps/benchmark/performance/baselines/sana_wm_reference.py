#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict SANA-WM reference benchmark used by the performance application."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--image", required=True, type=Path)
    value.add_argument("--model-dir", required=True, type=Path)
    value.add_argument("--prompt", required=True, type=Path)
    value.add_argument("--action", required=True)
    value.add_argument("--intrinsics", required=True, type=Path)
    value.add_argument("--num_frames", required=True, type=int)
    value.add_argument("--fps", required=True, type=int)
    value.add_argument("--step", required=True, type=int)
    value.add_argument("--cfg_scale", required=True, type=float)
    value.add_argument("--flow_shift", required=True, type=float)
    value.add_argument("--seed", required=True, type=int)
    value.add_argument("--translation_speed", type=float)
    value.add_argument("--rotation_speed_deg", type=float)
    value.add_argument("--output_dir", type=Path)
    value.add_argument("--no_action_overlay", action="store_true")
    return value


def prompt_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".json":
        value = json.loads(text)
        text = str(value.get("prompt", ""))
    if not text:
        raise RuntimeError("SANA-WM prompt is empty")
    return text


def media_summary(result) -> dict[str, int | str]:
    import numpy as np

    frames = getattr(result, "frames", None)
    if frames is None:
        raise RuntimeError("SANA-WM reference returned no materialized video frames")
    shape = tuple(int(value) for value in getattr(frames, "shape", ()))
    if len(shape) == 5:
        _, frame_count, first, second, third = shape
        if third in {1, 3, 4}:
            height, width, channels = first, second, third
        elif first in {1, 3, 4}:
            channels, height, width = first, second, third
        else:
            raise RuntimeError(f"unsupported SANA-WM video tensor shape: {shape}")
    elif len(shape) == 4:
        frame_count, first, second, third = shape
        if third in {1, 3, 4}:
            height, width, channels = first, second, third
        elif first in {1, 3, 4}:
            channels, height, width = first, second, third
        else:
            raise RuntimeError(f"unsupported SANA-WM video tensor shape: {shape}")
    else:
        video = frames[0]
        if not isinstance(video, (list, tuple)) or not video:
            raise RuntimeError("SANA-WM frames must contain one non-empty materialized video")
        frame_count = len(video)
        first_frame = video[0]
        if hasattr(first_frame, "size") and not isinstance(first_frame, np.ndarray):
            width, height = (int(value) for value in first_frame.size)
            channels = len(first_frame.getbands())
        else:
            frame_shape = tuple(int(value) for value in np.asarray(first_frame).shape)
            if len(frame_shape) != 3:
                raise RuntimeError(f"unsupported SANA-WM frame shape: {frame_shape}")
            if frame_shape[-1] in {1, 3, 4}:
                height, width, channels = frame_shape
            elif frame_shape[0] in {1, 3, 4}:
                channels, height, width = frame_shape
            else:
                raise RuntimeError(f"unsupported SANA-WM frame shape: {frame_shape}")
    return {
        "media_type": "video",
        "media_count": frame_count,
        "num_frames": frame_count,
        "height": height,
        "width": width,
        "channels": channels,
    }


def main() -> int:
    arguments = parser().parse_args()
    output_path = os.environ.get("TRTMC_SANA_WM_BENCHMARK_OUTPUT")
    if not output_path:
        raise RuntimeError("TRTMC_SANA_WM_BENCHMARK_OUTPUT is required")
    import numpy as np
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image

    pipeline = DiffusionPipeline.from_pretrained(
        arguments.model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    parameters = inspect.signature(pipeline.__call__).parameters
    required = {"prompt", "image", "action", "camera_intrinsics", "num_frames"}
    if not required.issubset(parameters):
        raise RuntimeError("installed SANA-WM pipeline lacks the required camera-control API")

    image = Image.open(arguments.image).convert("RGB")
    intrinsics = np.load(arguments.intrinsics).tolist()
    prompt = prompt_text(arguments.prompt)
    def invoke():
        generator = torch.Generator(device="cuda").manual_seed(arguments.seed)
        return pipeline(
            prompt=prompt,
            image=image,
            action=arguments.action,
            camera_intrinsics=intrinsics,
            num_frames=arguments.num_frames,
            num_inference_steps=arguments.step,
            guidance_scale=arguments.cfg_scale,
            generator=generator,
        )

    warmup = int(os.environ.get("TRTMC_SANA_WM_BENCHMARK_WARMUP", "1"))
    iterations = int(os.environ.get("TRTMC_SANA_WM_BENCHMARK_ITERATIONS", "1"))
    for _ in range(warmup):
        invoke()
    samples = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = invoke()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    summary = media_summary(result)
    Path(output_path).write_text(
        json.dumps(
            {
                "samples_ms": samples,
                "output_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
