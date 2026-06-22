"""phi4_multimodal model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.vision_language import VisionLanguageRunner


class Phi4MultimodalVisionLanguageGenerationRunner(VisionLanguageRunner):
    """phi4_multimodal local runner for vision_language_generation."""

runner = Phi4MultimodalVisionLanguageGenerationRunner()
