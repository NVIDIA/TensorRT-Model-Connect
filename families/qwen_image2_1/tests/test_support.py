# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and default-task policy for Qwen-Image 2.1."""

from families.qwen_image2_1.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_pipeline_class_selects_text_to_image() -> None:
    support = describe(ModelMetadata(config={}, model_index={"_class_name": "QwenImage21Pipeline"}))
    assert support is not None
    assert support.tasks == ("text_to_image",)
    assert support.default_task == "text_to_image"
    assert support.default_precision == "bf16"


def test_checkpoint_resolves_to_exactly_this_family() -> None:
    family, support = resolve_family(
        ModelMetadata(config={}, model_index={"_class_name": "QwenImage21Pipeline"})
    )
    assert family == "qwen_image2_1"
    assert support.default_task == "text_to_image"


def test_other_qwen_image_generations_are_not_claimed() -> None:
    for pipeline in ("QwenImagePipeline", "QwenImageEditPipeline", "QwenImageEditPlusPipeline"):
        assert describe(ModelMetadata(config={}, model_index={"_class_name": pipeline})) is None
    # The text encoder's own identity belongs to the vision-language family, not to this one.
    assert describe(ModelMetadata(config={"model_type": "qwen3_vl"}, model_index={})) is None
    assert describe(ModelMetadata(config={}, model_index={})) is None
