# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gpt2 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class Gpt2TextGenerationCausalComparator(TextComparator):
    """gpt2 local comparator for text_generation_causal."""

comparator = Gpt2TextGenerationCausalComparator()
