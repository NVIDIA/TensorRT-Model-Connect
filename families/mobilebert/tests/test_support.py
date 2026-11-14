# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for MobileBERT."""

from families.mobilebert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _metadata(**config) -> ModelMetadata:
    return ModelMetadata(config=config, model_index={})


def test_both_pretraining_heads_are_offered_with_masked_lm_first():
    support = describe(_metadata(model_type="mobilebert",
                                 architectures=["MobileBertForPreTraining"]))
    assert support is not None
    assert support.tasks == ("masked_text_to_token_scores",
                             "text_pair_to_pretraining_relation_scores")
    assert support.default_task == "masked_text_to_token_scores"
    assert support.default_precision == "fp32"


def test_the_bert_shaped_neighbours_are_not_claimed():
    for model_type in ("bert", "albert", "distilbert", "electra", "convbert", "roberta",
                       "esm", "mobilevit"):
        assert describe(_metadata(model_type=model_type)) is None
