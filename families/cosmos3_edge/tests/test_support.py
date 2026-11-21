# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The unified provider snapshot must have exactly one Model Connect owner."""

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("pipeline", [
    "Cosmos3OmniPipeline", "Cosmos3OmniDiffusersPipeline", "Cosmos3OmniModularPipeline",
])
def test_reasoner_is_not_claimed_by_generation_owner(pipeline):
    name, support = resolve_family(ModelMetadata(
        config={"model_type": "cosmos3_edge", "architectures": ["Cosmos3EdgeForConditionalGeneration"]},
        model_index={"_class_name": pipeline},
    ))
    assert name == "cosmos3_edge"
    assert support.default_task == "vision_language_generation"
    assert "image_video_text_to_text" in support.tasks
    assert "image_generation" not in support.tasks


def test_existing_cosmos3_nano_generation_discovery_is_unchanged():
    name, support = resolve_family(ModelMetadata(
        config={"model_type": "cosmos3_nano"},
        model_index={"_class_name": "Cosmos3OmniDiffusersPipeline"},
    ))
    assert name == "cosmos3"
    assert support.tasks == ("image_generation",)
