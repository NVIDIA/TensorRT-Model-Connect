# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for bart."""

from families.bart.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={"model_type": "bart"}, model_index={}))
    assert support is not None
    assert support.tasks == ("conditional_text_generation", "text_generation")
    assert support.default_task == "conditional_text_generation"


def test_mbart_model_type_supported():
    support = describe(ModelMetadata(config={"model_type": "mbart"}, model_index={}))
    assert support is not None
    assert support.tasks == ("conditional_text_generation", "text_generation")
    assert support.default_task == "conditional_text_generation"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"model_type": "unrelated-model"}, model_index={})) is None

