# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for GTE-v1.5."""

from families.gte.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_text_embedding_is_offered():
    # "new" is the literal model_type the GTE-v1.5 configs ship with.
    support = describe(ModelMetadata(config={"model_type": "new"}, model_index={}))
    assert support is not None
    # The checkpoint is an encoder with no head on top of it, so there is
    # nothing else honest to offer.
    assert support.tasks == ("text_to_embedding",)
    assert support.default_task == "text_to_embedding"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["NewModel"]}, model_index={}))
    assert support is not None
    assert support.default_task == "text_to_embedding"


def test_other_text_encoders_are_not_claimed():
    # The BERT-shaped families next door cover these. This checkpoint shares
    # their tokenizer and almost nothing else: rotary positions instead of a
    # table, a gated gelu MLP, and no token type embedding at all.
    for model_type in ("bert", "roberta", "distilbert", "electra", "mpnet", "convbert",
                       "nomic_bert", "modernbert"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_the_heads_on_this_encoder_are_not_claimed():
    # The remote code also defines masked-LM, classification, multiple-choice
    # and question-answering heads over the same encoder. None of them is an
    # embedding model, and this family builds no head at all.
    for architecture in ("NewForMaskedLM", "NewForSequenceClassification",
                         "NewForTokenClassification", "NewForQuestionAnswering",
                         "NewForMultipleChoice"):
        support = describe(ModelMetadata(config={"architectures": [architecture]}, model_index={}))
        assert support is None or support.tasks == ("text_to_embedding",)
