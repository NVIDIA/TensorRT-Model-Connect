# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gpt_neox model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class GptNeoxTextGenerationCausalComparator(TextComparator):
    """gpt_neox local comparator for text_generation_causal."""

comparator = GptNeoxTextGenerationCausalComparator()
