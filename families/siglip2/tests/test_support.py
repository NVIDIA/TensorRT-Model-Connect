# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for SigLIP 2 NaFlex."""

from families.siglip2.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_three_tasks_are_offered():
    support = describe(ModelMetadata(config={"model_type": "siglip2"}, model_index={}))
    assert support is not None
    assert support.tasks == ("image_to_embedding", "text_to_embedding",
                             "text_image_to_relevance")
    assert support.default_task == "text_image_to_relevance"
    assert support.default_precision == "fp32"


def test_architecture_also_resolves():
    support = describe(ModelMetadata(config={"architectures": ["Siglip2Model"]}, model_index={}))
    assert support is not None
    assert support.default_task == "text_image_to_relevance"


def test_fixed_resolution_siglip2_checkpoints_are_not_claimed():
    # The fixed-resolution SigLIP 2 checkpoints publish model_type "siglip" and the
    # SiglipModel graph: they belong to the siglip family, not to this one.
    assert describe(ModelMetadata(config={"model_type": "siglip",
                                          "architectures": ["SiglipModel"]},
                                  model_index={})) is None


def test_the_neighbouring_towers_are_not_claimed():
    # Single towers and the CLIP family share the two-tower naming but not this
    # checkpoint's identity.
    for model_type in ("siglip2_vision_model", "siglip2_text_model", "siglip_vision_model",
                       "clip", "lfm2_vl"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
