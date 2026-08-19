# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mixtral model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class MixtralTextGenerationCausalComparator(TextComparator):
    """mixtral local comparator for text_generation_causal."""

comparator = MixtralTextGenerationCausalComparator()
