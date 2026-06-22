"""flux model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class FluxDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """flux local runner for diffusion_media_generation."""

runner = FluxDiffusionMediaGenerationRunner()
