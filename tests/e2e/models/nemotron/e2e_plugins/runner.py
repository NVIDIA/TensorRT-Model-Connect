"""nemotron model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class NemotronTextGenerationCausalRunner(TextGenerationCausalRunner):
    """nemotron local runner for text_generation_causal."""

runner = NemotronTextGenerationCausalRunner()
