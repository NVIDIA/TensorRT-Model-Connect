# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_8 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class Qwen38TextGenerationCausalComparator(TextComparator):
    """qwen3_8 local comparator for text_generation_causal."""

comparator = Qwen38TextGenerationCausalComparator()
