# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from tensorrt_model_connect.model_support import ModelMetadata
from families.qwen3_vl_moe.support import describe


def test_moe_owner_and_default():
    support = describe(
        ModelMetadata(
            config={
                "model_type": "qwen3_vl_moe",
                "architectures": ["Qwen3VLMoeForConditionalGeneration"],
            },
            model_index={},
        )
    )
    assert support.tasks == ("vision_language_generation",)
    assert support.default_precision == "bf16"


@pytest.mark.parametrize(
    "kind,arch",
    [
        ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen3_vl_moe", "Qwen3VLForConditionalGeneration"),
    ],
)
def test_other_owners_are_not_claimed(kind, arch):
    assert (
        describe(
            ModelMetadata(config={"model_type": kind, "architectures": [arch]}, model_index={})
        )
        is None
    )
