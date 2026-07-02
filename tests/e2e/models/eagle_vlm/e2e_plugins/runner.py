# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""eagle_vlm model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.embedding import EmbeddingRunner
from .runners.reranking import RerankingRunner


class EagleVlmEmbeddingRunner(EmbeddingRunner):
    """eagle_vlm local runner for embedding."""


class EagleVlmRerankingRunner(RerankingRunner):
    """eagle_vlm local runner for reranking."""

runner = [
    EagleVlmEmbeddingRunner(),
    EagleVlmRerankingRunner(),
]
