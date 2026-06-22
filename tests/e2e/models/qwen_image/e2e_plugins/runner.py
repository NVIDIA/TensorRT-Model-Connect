"""qwen_image model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion import DiffusionMediaRunner


class QwenImageDiffusionMediaGenerationRunner(DiffusionMediaRunner):
    """qwen_image local runner for diffusion_media_generation."""

runner = QwenImageDiffusionMediaGenerationRunner()
