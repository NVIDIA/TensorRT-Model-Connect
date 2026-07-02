# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""eagle_vlm model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.embedding import EmbeddingComparator
from .comparators.reranking import RerankingComparator


class EagleVlmEmbeddingComparator(EmbeddingComparator):
    """eagle_vlm local comparator for embedding."""


class EagleVlmRerankingComparator(RerankingComparator):
    """eagle_vlm local comparator for reranking."""

comparator = [
    EagleVlmEmbeddingComparator(),
    EagleVlmRerankingComparator(),
]
