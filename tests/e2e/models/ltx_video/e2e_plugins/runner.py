"""ltx_video model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class LtxVideoDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """ltx_video local runner for diffusion_media_generation."""

runner = LtxVideoDiffusionMediaGenerationRunner()
