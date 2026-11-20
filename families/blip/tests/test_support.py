# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.blip.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_captioning_support():
    support = describe(
        ModelMetadata(
            config={"model_type": "blip", "architectures": ["BlipForConditionalGeneration"]},
            model_index={},
        )
    )
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp32"


@pytest.mark.parametrize(
    "kind,architecture",
    [
        ("blip", "BlipForQuestionAnswering"),
        ("blip", "BlipForImageTextRetrieval"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("blip", "BlipModel"),
        ("other", "BlipForConditionalGeneration"),
        ("blip", ""),
    ],
)
def test_other_tasks_are_not_claimed(kind, architecture):
    assert (
        describe(
            ModelMetadata(
                config={"model_type": kind, "architectures": [architecture]}, model_index={}
            )
        )
        is None
    )
