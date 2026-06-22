"""internlm model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class InternlmTextGenerationCausalRunner(TextGenerationCausalRunner):
    """internlm local runner for text_generation_causal."""

runner = InternlmTextGenerationCausalRunner()
