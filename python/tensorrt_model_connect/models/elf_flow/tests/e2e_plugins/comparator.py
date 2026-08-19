# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""elf_flow model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.diffusion_text_generation import DiffusionTextGenerationComparator


class ElfFlowDiffusionTextGenerationComparator(DiffusionTextGenerationComparator):
    """elf_flow local comparator for diffusion_text_generation."""

comparator = ElfFlowDiffusionTextGenerationComparator()
