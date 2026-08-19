# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-native Qwen-VL prompt framing contracts."""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "tensorrt",
    reason="Qwen-VL prompt tests import the TensorRT-backed family plugin",
)

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.models.qwen_vl import model as QwenVLModel
_QWEN25_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|>{image_pads}<|vision_end|>{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_QWEN3_TEMPLATE = (
    "<|im_start|>user\n"
    "<|vision_start|>{image_pads}<|vision_end|>{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.parametrize(
    ("model_type", "patch_size", "expected_pad_count", "expected_template"),
    (
        ("qwen2_vl", 14, 256, _QWEN25_TEMPLATE),
        ("qwen3_vl", 16, 196, _QWEN3_TEMPLATE),
    ),
)
def test_vl_prompt_template_matches_checkpoint_chat_contract(
    model_type: str,
    patch_size: int,
    expected_pad_count: int,
    expected_template: str,
) -> None:
    config = ModelConfig.from_json(
        json.dumps(
            {
                "model_type": model_type,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "vision_config": {
                    "patch_size": patch_size,
                    "spatial_merge_size": 2,
                    "deepstack_visual_indexes": (
                        [0] if model_type == "qwen3_vl" else []
                    ),
                },
            }
        )
    )

    vl_config = QwenVLModel.get_vl_config(config)

    assert vl_config is not None
    assert vl_config["num_image_pad_tokens"] == expected_pad_count
    assert vl_config["vl_prompt_template"] == expected_template
    rendered = expected_template.format(
        image_pads="<|image_pad|>" * expected_pad_count,
        prompt="What is shown?",
    )
    assert rendered.count("<|image_pad|>") == expected_pad_count
    assert rendered.index("<|vision_start|>") < rendered.index("What is shown?")
