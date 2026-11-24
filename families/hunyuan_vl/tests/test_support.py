# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunYuanVL checkpoints have exactly one owner, and this family claims nothing else."""

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from families.hunyuan_vl.support import describe


def test_hunyuan_vl_checkpoint_resolves_to_this_family():
    name, support = resolve_family(
        ModelMetadata(
            config={
                "model_type": "hunyuan_vl",
                "architectures": ["HunYuanVLForConditionalGeneration"],
            },
            model_index={},
        )
    )
    assert name == "hunyuan_vl"
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "bf16"


def test_dense_hunyuan_text_checkpoint_is_not_claimed():
    support = describe(
        ModelMetadata(
            config={
                "model_type": "hunyuan_v1_dense",
                "architectures": ["HunYuanDenseV1ForCausalLM"],
            },
            model_index={},
        )
    )
    assert support is None
