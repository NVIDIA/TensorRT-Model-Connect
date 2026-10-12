# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for timm XCiT."""

from families.timm_xcit.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_primary_task_matches_the_semantic_runtime():
    support = describe(ModelMetadata(config={"architecture": "xcit_large_24_p16_224"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_class_scores",)
    assert support.default_task == "image_to_class_scores"


def test_unrelated_identity_is_not_claimed():
    assert describe(ModelMetadata(config={"architecture": "unrelated-model"}, model_index={})) is None
