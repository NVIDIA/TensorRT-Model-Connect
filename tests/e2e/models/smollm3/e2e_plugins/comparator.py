# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""smollm3 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class SmolLM3TextGenerationCausalComparator(TextComparator):
    """smollm3 local comparator for text_generation_causal."""

comparator = SmolLM3TextGenerationCausalComparator()
