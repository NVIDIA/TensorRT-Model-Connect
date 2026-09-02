# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bert model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class BertEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """bert local runner for encoder_only_nlp."""


class BertEmbeddingRunner(EncoderOnlyRunner):
    """BERT local runner for pooled embedding checkpoints."""

    @property
    def strategy_name(self) -> str:
        return "embedding"


runner = [BertEncoderOnlyNlpRunner(), BertEmbeddingRunner()]
