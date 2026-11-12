# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity and Task policy for minicpmv4_7."""

from families.minicpmv4_7.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_checkpoint_identity_selects_images_text_to_text():
    support = describe(ModelMetadata(
        config={"model_type": "minicpmv4_7", "architectures": ["MiniCPMV4_7ForConditionalGeneration"]},
        model_index={}))
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "bf16"


def test_neighbouring_minicpm_generations_are_not_claimed():
    for model_type in ("minicpmv4_6", "minicpmv", "minicpm_ocr", "minicpm", "qwen3_5_moe"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_missing_model_type_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={})) is None
