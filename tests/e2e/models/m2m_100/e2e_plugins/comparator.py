# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""m2m_100 model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class M2m100TextGenerationCausalComparator(TextComparator):
    """m2m_100 local comparator for text_generation_causal."""

comparator = M2m100TextGenerationCausalComparator()
