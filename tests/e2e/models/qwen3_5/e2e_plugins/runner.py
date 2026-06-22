"""qwen3_5 model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class Qwen35TextGenerationCausalRunner(TextGenerationCausalRunner):
    """qwen3_5 local runner for text_generation_causal."""

runner = Qwen35TextGenerationCausalRunner()
