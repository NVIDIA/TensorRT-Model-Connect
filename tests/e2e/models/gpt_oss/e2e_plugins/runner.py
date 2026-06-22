"""gpt_oss model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class GptOssTextGenerationCausalRunner(TextGenerationCausalRunner):
    """gpt_oss local runner for text_generation_causal."""

runner = GptOssTextGenerationCausalRunner()
