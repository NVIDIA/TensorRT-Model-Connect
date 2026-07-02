# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen_image model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion import DiffusionComparator


class QwenImageDiffusionMediaGenerationComparator(DiffusionComparator):
    """qwen_image local comparator for diffusion_media_generation."""

comparator = QwenImageDiffusionMediaGenerationComparator()
