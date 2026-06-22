"""marian model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class MarianTextGenerationCausalRunner(TextGenerationCausalRunner):
    """marian local runner for text_generation_causal."""

runner = MarianTextGenerationCausalRunner()
