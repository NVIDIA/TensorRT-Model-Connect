# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna identity and default task, discoverable without TensorRT."""

from families.laguna.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def test_laguna_checkpoint_identity_and_default_task() -> None:
    config = {"model_type": "laguna", "architectures": ["LagunaForCausalLM"]}
    support = describe(ModelMetadata(config=config, model_index={}))
    assert support is not None
    assert support.default_task == "text_continuation"
    assert support.tasks == ("text_continuation",)
    assert support.default_precision == "bf16"


def test_architecture_alone_identifies_laguna() -> None:
    support = describe(
        ModelMetadata(config={"architectures": ["LagunaForCausalLM"]}, model_index={})
    )
    assert support is not None and support.default_task == "text_continuation"


def test_other_checkpoints_are_not_laguna() -> None:
    for config in (
        {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]},
        {"model_type": "afmoe", "architectures": ["AfmoeForCausalLM"]},
        {"model_type": "", "architectures": []},
    ):
        assert describe(ModelMetadata(config=config, model_index={})) is None
