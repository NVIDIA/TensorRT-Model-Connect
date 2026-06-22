"""m2m_100 model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.text_generation import TextGenerationCausalRunner


class M2m100TextGenerationCausalRunner(TextGenerationCausalRunner):
    """m2m_100 local runner for text_generation_causal."""

runner = M2m100TextGenerationCausalRunner()
