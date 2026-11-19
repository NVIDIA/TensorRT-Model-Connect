# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.blip2.support import describe


def test_blip2_checkpoint_has_one_owner():
    family, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "blip-2",
                "architectures": ["Blip2ForConditionalGeneration"],
            },
            {},
        )
    )
    assert family == "blip2"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"


def test_standalone_vision_checkpoint_is_not_claimed():
    assert (
        describe(
            ModelMetadata(
                {"model_type": "blip_2_vision_model", "architectures": ["Blip2VisionModel"]}, {}
            )
        )
        is None
    )
