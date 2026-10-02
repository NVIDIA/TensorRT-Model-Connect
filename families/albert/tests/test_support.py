# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for albert."""

from families.albert.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_is_text_to_pooled_features():
    support = describe(ModelMetadata(config={"model_type": "albert"}, model_index={}))
    assert support is not None
    assert support.default_task == "text_to_pooled_features"


def test_all_semantic_tasks_declared():
    support = describe(ModelMetadata(config={"model_type": "albert"}, model_index={}))
    assert support is not None
    assert "text_to_pooled_features" in support.tasks
    assert "text_to_token_features" in support.tasks
    assert "text_to_embedding" in support.tasks
    assert "text_pair_to_relevance" in support.tasks


def test_unrelated_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "bert"}, model_index={})) is None


def test_missing_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={})) is None
