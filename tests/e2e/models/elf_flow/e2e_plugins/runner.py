# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""elf_flow model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.diffusion_text_generation import DiffusionTextGenerationRunner


class ElfFlowDiffusionTextGenerationRunner(DiffusionTextGenerationRunner):
    """elf_flow local runner for diffusion_text_generation."""

runner = ElfFlowDiffusionTextGenerationRunner()
