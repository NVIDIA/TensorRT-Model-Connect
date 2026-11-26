# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint identity and task policy for MiniCPM-V 4.6."""

from families.minicpmv4_6.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_checkpoint_identity_selects_images_text_to_text():
    support = describe(ModelMetadata(
        config={"model_type": "minicpmv4_6", "architectures": ["MiniCPMV4_6ForConditionalGeneration"]},
        model_index={}))
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp16"


def test_architecture_alone_resolves():
    support = describe(ModelMetadata(
        config={"architectures": ["MiniCPMV4_6ForConditionalGeneration"]}, model_index={}))
    assert support is not None
    assert support.default_task == "images_text_to_text"


def test_neighbouring_generations_and_towers_are_not_claimed():
    # Other MiniCPM-V generations differ in tower, merger or decoder; the
    # Qwen3.5 decoder and the vision tower alone are not this model either.
    for model_type in ("minicpmv4_7", "minicpmv", "minicpm_ocr", "minicpmo", "minicpm",
                       "minicpmv4_6_vision", "qwen3_5", "qwen3_5_text"):
        assert describe(ModelMetadata(config={"model_type": model_type}, model_index={})) is None


def test_missing_identity_is_not_claimed():
    assert describe(ModelMetadata(config={}, model_index={})) is None
