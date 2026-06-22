"""gpt2 model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class Gpt2TextGenerationCausalRunner(TextGenerationCausalRunner):
    """gpt2 local runner for text_generation_causal."""

runner = Gpt2TextGenerationCausalRunner()
