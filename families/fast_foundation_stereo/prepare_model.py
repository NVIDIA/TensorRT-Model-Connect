# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose the official HF checkpoint with its pinned open-source model code."""

from __future__ import annotations


def configure_official_model_args(
    model,
    *,
    max_disparity: int,
    valid_iters: int,
) -> None:
    """Apply the complete inference contract omitted by older checkpoints."""
    model.args.max_disp = max_disparity
    model.args.valid_iters = valid_iters
    model.args.normalize = True
