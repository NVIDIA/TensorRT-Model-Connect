# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for CLIP."""

from families.clip.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_all_three_tasks_are_offered():
    support = describe(ModelMetadata(config={"model_type": "clip"}, model_index={}))
    assert support is not None
    # Two towers that meet at a dot product: each can be asked for its own
    # embedding, and the pair can be scored. There is no joint
    # image_text_to_embedding here - CLIP has no encoder that takes both.
    assert support.tasks == ("image_to_embedding", "text_to_embedding",
                             "text_image_to_relevance")
    assert support.default_task == "text_image_to_relevance"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["CLIPModel"]}, model_index={})
    )
    assert support is not None
    assert support.default_task == "text_image_to_relevance"


def test_unrelated_identity_is_not_claimed():
    # CLIP ships inside several vision-language checkpoints as a sub-encoder;
    # this family claims the standalone model only.
    assert describe(ModelMetadata(config={"model_type": "siglip"}, model_index={})) is None
