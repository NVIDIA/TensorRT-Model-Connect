# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for SigLIP."""

from families.siglip.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_three_tasks_are_offered():
    support = describe(ModelMetadata(config={"model_type": "siglip"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_embedding", "text_to_embedding",
                             "text_image_to_relevance")
    assert support.default_task == "text_image_to_relevance"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["SiglipModel"]}, model_index={}))
    assert support is not None
    assert support.default_task == "text_image_to_relevance"


def test_siglip2_is_owned_by_the_siglip_family():
    support = describe(ModelMetadata(config={"model_type": "siglip"}, model_index={}))
    assert support is not None


def test_the_neighbouring_towers_are_not_claimed():
    # CLIP and its descendants share the two-tower shape and much of the
    # tensor naming, but none of them has the attention-pooling head or the
    # sigmoid scoring, so none may be silently claimed by this family.
    for model_type in ("clip", "siglip_vision_model", "altclip", "chinese_clip", "owlv2"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
