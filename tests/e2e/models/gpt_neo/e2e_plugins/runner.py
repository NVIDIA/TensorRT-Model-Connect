"""gpt_neo model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class GptNeoTextGenerationCausalRunner(TextGenerationCausalRunner):
    """gpt_neo local runner for text_generation_causal."""

runner = GptNeoTextGenerationCausalRunner()
