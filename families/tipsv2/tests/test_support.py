# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for TIPSv2."""

from families.tipsv2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_three_tasks_are_offered():
    support = describe(ModelMetadata(config={"model_type": "tipsv2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_embedding", "text_to_embedding",
                             "text_image_to_relevance")
    assert support.default_task == "text_image_to_relevance"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["TIPSv2Model"]}, model_index={}))
    assert support is not None
    assert support.default_task == "text_image_to_relevance"


def test_the_dense_prediction_sibling_is_not_claimed():
    # google/tipsv2-*-dpt reuses these towers under depth, normal and segmentation heads. It is a
    # different model_type and a different set of Tasks, so claiming it here would offer a
    # relevance score for a checkpoint that has no text tower to score with.
    assert describe(ModelMetadata(config={"model_type": "tipsv2_dpt"}, model_index={})) is None
    assert describe(ModelMetadata(config={"architectures": ["Tipsv2DptForDepthEstimation"]},
                                  model_index={})) is None


def test_the_neighbouring_towers_are_not_claimed():
    # CLIP, SigLIP and DINOv2 share pieces of this shape - a two-tower contrastive model, or a ViT
    # with register tokens - but none of them has TIPSv2's combination of a DINOv2 vision tower
    # with a masked-average-pooled text tower, so none may be silently claimed here.
    for model_type in ("clip", "siglip", "siglip2", "chinese_clip", "dinov2", "dinov3", "owlv2"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
