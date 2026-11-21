# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Existing Gemma identity regression, discoverable without TensorRT."""

from families.gemma.support import LEGACY_TEXT_MODEL_TYPES, describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_the_gate_matches_what_support_claims() -> None:
    """The builder and identity check use the same legacy type declaration."""
    for model_type in LEGACY_TEXT_MODEL_TYPES | {"gemma4_text", "gemma4_unified_text"}:
        support = describe(ModelMetadata(config={"model_type": model_type}, model_index={}))
        assert support is not None
        assert support.default_task == "text_generation"
    for model_type in ("gemma4", "gemma4_unified"):
        for media, expected in (({}, "text_continuation"),
                                ({"vision_config": {}}, "images_text_to_text"),
                                ({"audio_config": {}}, "audio_text_to_text"),
                                ({"vision_config": {}, "audio_config": {}}, "images_text_to_text")):
            support = describe(ModelMetadata(config={"model_type": model_type, **media}, model_index={}))
            assert support is not None
            assert support.default_task == expected
            assert expected in support.tasks
