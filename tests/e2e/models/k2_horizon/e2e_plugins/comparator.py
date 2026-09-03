# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""K2-Horizon model-owned comparator registration."""

from __future__ import annotations

from .comparators.text import TextComparator


class K2HorizonTextGenerationCausalComparator(TextComparator):
    """Strict K2-Horizon causal-generation comparator."""


comparator = K2HorizonTextGenerationCausalComparator()
