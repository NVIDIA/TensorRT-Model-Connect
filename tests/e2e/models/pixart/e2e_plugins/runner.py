"""pixart model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class PixartDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """pixart local runner for diffusion_media_generation."""

runner = PixartDiffusionMediaGenerationRunner()
