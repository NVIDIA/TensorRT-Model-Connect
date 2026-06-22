"""qwen_moe model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class QwenMoeTextGenerationCausalRunner(TextGenerationCausalRunner):
    """qwen_moe local runner for text_generation_causal."""

runner = QwenMoeTextGenerationCausalRunner()
