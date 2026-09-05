#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official SANA-WM pipeline for the performance application."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--reference-repo", required=True, type=Path)
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
    value.add_argument("--refiner_seed", required=True, type=int)
    value.add_argument("--translation_speed", required=True, type=float)
    value.add_argument("--rotation_speed_deg", required=True, type=float)
    value.add_argument("--no_action_overlay", action="store_true")
    value.add_argument("--warmup", required=True, type=int)
    value.add_argument("--iterations", required=True, type=int)
    value.add_argument("--output", required=True, type=Path)
    return value


def media_summary(video: Any) -> dict[str, int | str]:
    import numpy as np

    shape = tuple(int(value) for value in np.asarray(video).shape)
    if len(shape) != 4 or shape[-1] not in {1, 3, 4}:
        raise RuntimeError(f"official SANA-WM video must be THWC, got {shape}")
    frames, height, width, channels = shape
    if frames < 1:
        raise RuntimeError("official SANA-WM returned no video frames")
    return {
        "media_type": "video",
        "media_count": frames,
        "num_frames": frames,
        "height": height,
        "width": width,
        "channels": channels,
    }


def _official_module(reference_repo: Path) -> Any:
    entrypoint = reference_repo / "inference_video_scripts/wm/inference_sana_wm.py"
    if not entrypoint.is_file():
        raise RuntimeError(f"official SANA-WM entrypoint does not exist: {entrypoint}")
    sys.path.insert(0, str(reference_repo))
    from inference_video_scripts.wm import inference_sana_wm

    return inference_sana_wm


def main() -> int:
    arguments = parser().parse_args()
    if arguments.warmup < 0 or arguments.iterations < 1:
        raise ValueError("SANA-WM benchmark warmup/iterations are invalid")

    import pyrallis
    import torch
    from PIL import Image

    reference_repo = arguments.reference_repo.resolve()
    model_dir = arguments.model_dir.resolve()
    official = _official_module(reference_repo)
    required_model_paths = {
        "config": model_dir / "config.yaml",
        "model": model_dir / "dit/sana_wm_1600m_720p.safetensors",
        "refiner": model_dir / "refiner",
        "refiner text encoder": model_dir / "refiner/text_encoder",
    }
    missing = [name for name, path in required_model_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError("SANA-WM model directory is missing: " + ", ".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("official SANA-WM performance requires CUDA")

    image = Image.open(arguments.image).convert("RGB")
    prompt = arguments.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError("SANA-WM prompt is empty")
    trajectory = official.action_string_to_c2w(
        arguments.action,
        translation_speed=arguments.translation_speed,
        rotation_speed_deg=arguments.rotation_speed_deg,
    )
    snapped_frames = official._snap_num_frames(
        arguments.num_frames,
        stride=8,
        upper_bound=int(trajectory.shape[0]),
    )
    if snapped_frames != arguments.num_frames:
        raise RuntimeError(
            f"official SANA-WM cannot execute exactly {arguments.num_frames} frames; "
            f"resolved {snapped_frames}"
        )
    trajectory = trajectory[: arguments.num_frames]
    cropped, source_size, resized_size, crop_offset = official.resize_and_center_crop(image)
    intrinsics = official.load_intrinsics(arguments.intrinsics, arguments.num_frames)
    intrinsics = official.transform_intrinsics_for_crop(
        intrinsics, source_size, resized_size, crop_offset
    )
    config = pyrallis.parse(
        config_class=official.InferenceConfig,
        config_path=required_model_paths["config"],
        args=[],
    )
    refiner = official.RefinerSettings(
        root=required_model_paths["refiner"],
        gemma_root=required_model_paths["refiner text encoder"],
        seed=arguments.refiner_seed,
    )
    pipeline = official.SanaWMPipeline(
        config=config,
        model_path=required_model_paths["model"],
        device="cuda",
        refiner=refiner,
    )
    params = official.GenerationParams(
        num_frames=arguments.num_frames,
        fps=arguments.fps,
        step=arguments.step,
        cfg_scale=arguments.cfg_scale,
        flow_shift=arguments.flow_shift,
        seed=arguments.seed,
    )

    def invoke() -> Any:
        return pipeline.generate(cropped, prompt, trajectory, intrinsics, params)

    for _ in range(arguments.warmup):
        invoke()
        torch.cuda.synchronize()
    samples = []
    result = None
    for _ in range(arguments.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = invoke()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    if not isinstance(result, dict) or "video" not in result:
        raise RuntimeError("official SANA-WM pipeline returned no video")
    video = result["video"]
    if not arguments.no_action_overlay:
        video = official.apply_overlay(video, result["c2w"])
    arguments.output.write_text(
        json.dumps(
            {"samples_ms": samples, "output_summary": media_summary(video)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
