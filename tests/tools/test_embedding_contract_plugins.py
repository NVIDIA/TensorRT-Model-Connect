# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for embedding and encoder contract threshold floors."""

from __future__ import annotations

import math

from tests.e2e.models.bert.e2e_plugins.contract import (
    BertEmbeddingPlugin,
    BertEncoderFeaturesPlugin,
)
from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile


def _outputs() -> tuple[StageOutput, StageOutput]:
    trt = StageOutput(
        stage_name="full_inference",
        data={"embedding": [1.0, 0.0]},
    )
    ref = StageOutput(
        stage_name="full_inference",
        data={"embedding": [0.7, math.sqrt(1.0 - 0.7 * 0.7)]},
    )
    return trt, ref


def test_embedding_contract_raises_unreasonably_low_cosine_threshold() -> None:
    trt, ref = _outputs()
    result = BertEmbeddingPlugin().verify(
        trt,
        ref,
        E2ECase(
            name="embedding-case",
            hf_id="hf/embedding-case",
            family="bert",
            runtime_strategy="bert_encoder_only",
            reference_family="sentence_transformer_embed",
        ),
        ThresholdProfile(
            task_strategy="encoder_only_nlp",
            metrics={"cls_embedding_cosine": -0.1},
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["cosine_similarity"].threshold == 0.8
    assert "raised" in result.metrics["cosine_similarity"].note


def test_encoder_contract_raises_unreasonably_low_cosine_threshold() -> None:
    trt, ref = _outputs()
    result = BertEncoderFeaturesPlugin().verify(
        trt,
        ref,
        E2ECase(
            name="encoder-case",
            hf_id="hf/encoder-case",
            family="bert",
            runtime_strategy="bert_encoder_only",
            reference_family="encoder_base_features",
        ),
        ThresholdProfile(
            task_strategy="encoder_only_nlp",
            metrics={"cls_embedding_cosine": 0.6},
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["cosine_similarity"].threshold == 0.8
