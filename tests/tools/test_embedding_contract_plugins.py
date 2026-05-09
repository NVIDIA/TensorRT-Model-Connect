"""Tests for embedding and encoder contract threshold floors."""

from __future__ import annotations

import math

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e_harness.plugins.embedding import EmbeddingPlugin
from tests.e2e_harness.plugins.encoder_features import EncoderFeaturesPlugin


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
    result = EmbeddingPlugin().verify(
        trt,
        ref,
        E2ECase(
            name="dpr",
            hf_id="hf/dpr",
            family="dpr",
            runtime_strategy="encoder_only",
            reference_family="dpr_context_embed",
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
    result = EncoderFeaturesPlugin().verify(
        trt,
        ref,
        E2ECase(
            name="albert",
            hf_id="hf/albert",
            family="albert",
            runtime_strategy="encoder_only",
            reference_family="encoder_base_features",
        ),
        ThresholdProfile(
            task_strategy="encoder_only_nlp",
            metrics={"cls_embedding_cosine": 0.6},
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["cosine_similarity"].threshold == 0.8
