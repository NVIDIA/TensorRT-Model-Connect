# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Middlebury ground-truth loading for this family's direct E2E."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def ground_truth(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    disparity_path = inputs.get("ground_truth_disparity")
    mask_path = inputs.get("valid_nonocc_mask")
    if not disparity_path or not mask_path:
        raise ValueError(
            "task-accuracy stereo inputs require ground_truth_disparity and valid_nonocc_mask"
        )
    disparity = np.load(Path(str(disparity_path)), allow_pickle=False).astype(
        np.float32, copy=False
    )
    valid = np.load(Path(str(mask_path)), allow_pickle=False).astype(bool, copy=False)
    if disparity.shape != (700, 700) or valid.shape != disparity.shape:
        raise ValueError("prepared stereo ground truth and mask must both have shape [700, 700]")
    if not valid.any():
        raise ValueError("prepared stereo valid non-occluded mask is empty")
    if not np.isfinite(disparity[valid]).all():
        raise ValueError("prepared stereo ground truth is non-finite on valid pixels")
    return disparity, valid
