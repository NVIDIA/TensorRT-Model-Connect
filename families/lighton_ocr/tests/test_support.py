# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery must keep OCR ownership separate from generic Mistral and Qwen."""

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from families.lighton_ocr.support import describe


@pytest.mark.parametrize("architecture", [
    "LightOnOCRForConditionalGeneration", "LightOnOcrForConditionalGeneration",
])
def test_legacy_checkpoint_resolves_to_its_ocr_owner(architecture):
    metadata = ModelMetadata(
        config={"model_type": "mistral3", "architectures": [architecture]}, model_index={},
    )
    family, support = resolve_family(metadata)
    assert family == "lighton_ocr"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp32"


def test_native_lighton_identity_is_supported():
    support = describe(ModelMetadata(config={"model_type": "lighton_ocr"}, model_index={}))
    assert support is not None and support.default_task == "images_text_to_text"


@pytest.mark.parametrize("config", [
    {},
    {"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"]},
    {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    {"model_type": "pixtral", "architectures": ["PixtralVisionModel"]},
    {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
])
def test_unrelated_models_are_not_claimed(config):
    assert describe(ModelMetadata(config=config, model_index={})) is None
