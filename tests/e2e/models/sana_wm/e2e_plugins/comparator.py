# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM model-owned diffusion comparator plugins."""

from __future__ import annotations

from .comparators.sana_wm import SanaWmComparator


class SanaWmDiffusionMediaGenerationComparator(SanaWmComparator):
    """SANA-WM local comparator for diffusion_media_generation."""

    @property
    def task_strategy(self) -> str:
        return "diffusion_media_generation"


comparator = SanaWmDiffusionMediaGenerationComparator()
