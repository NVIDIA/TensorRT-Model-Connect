# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-MoE model-owned comparator registration."""

from __future__ import annotations

from .comparators.text import TextComparator


class Lfm2MoeTextGenerationCausalComparator(TextComparator):
    """Strict LFM2-MoE causal-generation comparator."""


comparator = Lfm2MoeTextGenerationCausalComparator()
