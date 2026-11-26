# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma checkpoints with the Gemma decoder have exactly one owner."""

import pytest

from tensorrt_model_connect.model_support import (
    FamilyResolutionError,
    ModelMetadata,
    resolve_family,
)

PALIGEMMA_3B_PT_224 = {
    "architectures": ["PaliGemmaForConditionalGeneration"],
    "model_type": "paligemma",
    "text_config": {"model_type": "gemma"},
    "vision_config": {"model_type": "siglip_vision_model"},
}


def test_paligemma_resolves_to_its_family():
    name, support = resolve_family(ModelMetadata(config=PALIGEMMA_3B_PT_224, model_index={}))
    assert name == "paligemma"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "bf16"


def test_an_absent_text_model_type_means_gemma():
    config = dict(PALIGEMMA_3B_PT_224, text_config={"hidden_size": 2048})
    name, support = resolve_family(ModelMetadata(config=config, model_index={}))
    assert name == "paligemma"
    assert support.default_task == "images_text_to_text"


def test_paligemma_2_with_a_gemma_2_decoder_is_not_claimed():
    config = dict(PALIGEMMA_3B_PT_224, text_config={"model_type": "gemma2"})
    with pytest.raises(FamilyResolutionError, match="no family supports"):
        resolve_family(ModelMetadata(config=config, model_index={}))
