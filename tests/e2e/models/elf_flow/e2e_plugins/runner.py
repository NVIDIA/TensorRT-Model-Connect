"""elf_flow model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion_text_generation import DiffusionTextGenerationRunner


class ElfFlowDiffusionTextGenerationRunner(DiffusionTextGenerationRunner):
    """elf_flow local runner for diffusion_text_generation."""

runner = ElfFlowDiffusionTextGenerationRunner()
