# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and default Task of the OpenAI GPT family, without TensorRT."""

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_openai_gpt_lm_head_has_one_owner_and_continues_text():
    # config.json of openai-community/openai-gpt.
    metadata = ModelMetadata(
        config={"model_type": "openai-gpt", "architectures": ["OpenAIGPTLMHeadModel"]},
        model_index={},
        files=("config.json", "generation_config.json", "tokenizer.json", "model.safetensors"),
    )
    family, support = resolve_family(metadata)
    assert family == "openai_gpt"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
    assert support.default_precision == "fp32"


def test_openai_gpt_architecture_alone_resolves_to_the_family():
    family, support = resolve_family(
        ModelMetadata(config={"architectures": ["OpenAIGPTLMHeadModel"]}, model_index={})
    )
    assert family == "openai_gpt"
    assert support.default_task == "text_continuation"


def test_gpt2_stays_with_its_own_family():
    family, _ = resolve_family(
        ModelMetadata(
            config={"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}, model_index={}
        )
    )
    assert family == "gpt2"
