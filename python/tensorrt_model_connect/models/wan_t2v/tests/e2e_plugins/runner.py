# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""wan_t2v model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class WanT2vDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """wan_t2v local runner for diffusion_media_generation."""

runner = WanT2vDiffusionMediaGenerationRunner()
