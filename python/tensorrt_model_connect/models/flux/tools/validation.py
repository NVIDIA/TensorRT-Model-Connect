# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned validation refinements."""

from __future__ import annotations

from tensorrt_model_connect.models.flux.tests.e2e_plugins.comparators.clip_metrics import (
    compute_clip_metrics,
)


def compute_diffusion_metrics(
    trt_frames_dir: str,
    hf_frames_dir: str,
    prompt: str,
):
    """Compute the Flux CLIP metrics used by its validation suite."""

    return compute_clip_metrics(trt_frames_dir, hf_frames_dir, prompt)
