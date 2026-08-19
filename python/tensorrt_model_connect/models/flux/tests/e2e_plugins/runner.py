# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""flux model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class FluxDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """flux local runner for diffusion_media_generation."""

runner = FluxDiffusionMediaGenerationRunner()
