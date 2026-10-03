# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for distilbert."""

from families.distilbert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={"model_type": "distilbert"}, model_index={}))
    assert support is not None
    assert support.tasks == (
        "text_to_embedding",
        "text_to_pooled_features",
        "text_pair_to_relevance",
    )
    assert support.default_task == "text_to_pooled_features"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "unrelated-model"}, model_index={})) is None
