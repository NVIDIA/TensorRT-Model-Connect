# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free semantic Task discovery for GLM-OCR."""

from families.glm_ocr.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_images_text_to_text_is_offered():
    support = describe(ModelMetadata(config={"model_type": "glm_ocr"}, model_index={}))
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "bf16"


def test_architecture_also_resolves():
    support = describe(
        ModelMetadata(config={"architectures": ["GlmOcrForConditionalGeneration"]}, model_index={}))
    assert support is not None
    assert support.default_task == "images_text_to_text"


def test_the_other_glm_vision_layouts_are_not_claimed():
    # GLM-4.1V / GLM-4.5V / GLM-4.6V share the GLM-V lineage but differ in their vision
    # tower, connector and decoder; the text-only GLM and GLM-ASR have their own families.
    for model_type in ("glm4v", "glm4v_moe", "glm46v", "glm", "glm4", "glmasr", "glm_ocr_text",
                       "glm_ocr_vision"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None
    for architecture in ("Glm4vForConditionalGeneration", "GlmOcrModel", "GlmOcrTextModel"):
        assert describe(
            ModelMetadata(config={"architectures": [architecture]}, model_index={})) is None
