# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.paddleocr_vl.support import describe


def test_paddleocr_checkpoint_has_one_owner():
    family, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "paddleocr_vl",
                "architectures": ["PaddleOCRVLForConditionalGeneration"],
            },
            {},
        )
    )
    assert family == "paddleocr_vl"
    assert support.tasks == ("images_text_to_text",)


def test_standalone_vision_checkpoint_is_not_claimed():
    assert (
        describe(
            ModelMetadata(
                {"model_type": "paddleocr_vl", "architectures": ["PaddleOCRVisionModel"]}, {}
            )
        )
        is None
    )
