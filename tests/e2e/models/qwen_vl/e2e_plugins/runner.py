"""qwen_vl model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.vision_language import VisionLanguageRunner


class QwenVlVisionLanguageGenerationRunner(VisionLanguageRunner):
    """qwen_vl local runner for vision_language_generation."""

runner = QwenVlVisionLanguageGenerationRunner()
