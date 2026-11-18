# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_apertus_causal_lm_has_one_owner_and_continues_text():
    metadata = ModelMetadata(
        config={"model_type": "apertus", "architectures": ["ApertusForCausalLM"]},
        model_index={},
        files=(
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ),
    )
    family, support = resolve_family(metadata)
    assert family == "apertus"
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
    assert support.default_precision == "bf16"


def test_apertus_architecture_alone_resolves_to_the_family():
    family, support = resolve_family(
        ModelMetadata(config={"architectures": ["ApertusForCausalLM"]}, model_index={})
    )
    assert family == "apertus"
    assert support.default_task == "text_continuation"
