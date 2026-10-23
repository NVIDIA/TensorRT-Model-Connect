# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for nomic-embed-text."""

from families.nomic_bert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_text_embedding_is_offered():
    support = describe(ModelMetadata(config={"model_type": "nomic_bert"}, model_index={}))
    assert support is not None
    # The checkpoint is an encoder with no head on top of it, so there is
    # nothing else honest to offer.
    assert support.tasks == ("text_to_embedding",)
    assert support.default_task == "text_to_embedding"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["NomicBertModel"]}, model_index={}))
    assert support is not None
    assert support.default_task == "text_to_embedding"


def test_plain_bert_is_not_claimed():
    # families/bert covers the transformers BERT. This checkpoint shares its
    # tokenizer and almost nothing else: rotary positions instead of a table,
    # a SwiGLU MLP, and no biases anywhere.
    for model_type in ("bert", "roberta", "distilbert", "electra", "mpnet", "convbert"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None


def test_the_heads_on_this_encoder_are_not_claimed():
    # The remote code also defines classification and question-answering
    # heads over the same encoder. None of them is an embedding model, and
    # this family builds no head at all.
    for architecture in ("NomicBertForPreTraining", "NomicBertForSequenceClassification",
                         "NomicBertForQuestionAnswering", "NomicBertForTokenClassification"):
        support = describe(
            ModelMetadata(config={"architectures": [architecture]}, model_index={}))
        assert support is None or support.tasks == ("text_to_embedding",)
