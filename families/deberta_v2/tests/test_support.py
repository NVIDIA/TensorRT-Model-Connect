# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for DeBERTa-v2/v3."""

from families.deberta_v2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_token_features_are_offered():
    support = describe(ModelMetadata(config={"model_type": "deberta-v2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("text_to_token_features",)
    assert support.default_task == "text_to_token_features"


def test_both_spellings_of_the_model_type_resolve():
    # The config says "deberta-v2" with a hyphen; the underscore spelling is
    # what a reader expects from the directory name.
    for model_type in ("deberta-v2", "deberta_v2"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={}))


def test_deberta_v1_is_not_claimed():
    # families/deberta covers microsoft/deberta-base. It is a different
    # model_type with a different attention: v1 has no bucketed relative
    # positions and does not share its projections with the positional terms.
    assert describe(ModelMetadata(config={"model_type": "deberta"}, model_index={})) is None


def test_the_other_encoders_are_not_claimed():
    for model_type in ("bert", "roberta", "electra", "mpnet", "nomic_bert"):
        assert describe(ModelMetadata(config={"model_type": model_type},
                                      model_index={})) is None
