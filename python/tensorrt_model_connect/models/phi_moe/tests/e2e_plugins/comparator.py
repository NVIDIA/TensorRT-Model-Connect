# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""phi_moe model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text import TextComparator


class PhiMoeTextGenerationCausalComparator(TextComparator):
    """phi_moe local comparator for text_generation_causal."""

comparator = PhiMoeTextGenerationCausalComparator()
