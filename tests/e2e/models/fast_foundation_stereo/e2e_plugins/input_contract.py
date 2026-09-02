# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input and ground-truth helpers shared by the stereo model-owned plugins."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data"


def stage_stereo_inputs(directory: Path, inputs: Mapping[str, Any]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    configured = (inputs.get("left_image"), inputs.get("right_image"))
    if any(configured) and not all(configured):
        raise ValueError("prepared stereo inputs require both left_image and right_image")
    sources = (
        (Path(str(configured[0])), Path(str(configured[1])))
        if all(configured)
        else (_FIXTURE_DIR / "office_left.png", _FIXTURE_DIR / "office_right.png")
    )
    destinations = (directory / "left.png", directory / "right.png")
    for source, destination in zip(sources, destinations, strict=True):
        if not source.is_file():
            raise FileNotFoundError(f"stereo input does not exist: {source}")
        shutil.copyfile(source, destination)
    return destinations


def ground_truth(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    disparity_path = inputs.get("ground_truth_disparity")
    mask_path = inputs.get("valid_nonocc_mask")
    if disparity_path is None and mask_path is None:
        return None
    if not disparity_path or not mask_path:
        raise ValueError(
            "task-accuracy stereo inputs require ground_truth_disparity and valid_nonocc_mask"
        )
    disparity = np.load(Path(str(disparity_path)), allow_pickle=False).astype(
        np.float32, copy=False
    )
    valid = np.load(Path(str(mask_path)), allow_pickle=False).astype(bool, copy=False)
    if disparity.shape != (700, 700) or valid.shape != disparity.shape:
        raise ValueError(
            "prepared stereo ground truth and mask must both have shape [700, 700]"
        )
    if not valid.any():
        raise ValueError("prepared stereo valid non-occluded mask is empty")
    if not np.isfinite(disparity[valid]).all():
        raise ValueError("prepared stereo ground truth is non-finite on valid pixels")
    return disparity, valid


def attach_ground_truth(data: dict[str, Any], inputs: Mapping[str, Any]) -> None:
    task_truth = ground_truth(inputs)
    if task_truth is None:
        return
    disparity, valid = task_truth
    data.update(
        ground_truth_disparity=disparity,
        valid_nonocc_mask=valid,
        scene=str(inputs.get("scene", "")),
        preparation_transform=dict(inputs.get("transform", {})),
    )
