"""starcoder2 model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class Starcoder2TextGenerationCausalRunner(TextGenerationCausalRunner):
    """starcoder2 local runner for text_generation_causal."""

runner = Starcoder2TextGenerationCausalRunner()
