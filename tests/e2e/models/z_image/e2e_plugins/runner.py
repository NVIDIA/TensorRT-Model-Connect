# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""z_image model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class ZImageDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """z_image local runner for diffusion_media_generation."""

runner = ZImageDiffusionMediaGenerationRunner()
