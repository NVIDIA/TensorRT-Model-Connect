"""mistral model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class MistralTextGenerationCausalRunner(TextGenerationCausalRunner):
    """mistral local runner for text_generation_causal."""

runner = MistralTextGenerationCausalRunner()
