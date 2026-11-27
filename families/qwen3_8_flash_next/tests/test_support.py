# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.qwen3_8_flash_next.support import describe


def _prepared_target():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "ple_layer_ids": [2],
            "hidden_size": 2560,
        },
        "vision_config": {"out_hidden_size": 2560},
        "quantization_config": {
            "quant_method": "modelopt",
            "quantized_layers": {
                f"model.language_model.layers.{layer}.mlp.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                }
                for layer in range(48)
            },
        },
    }


def test_target_defaults_to_vision_language_generation():
    family, support = resolve_family(ModelMetadata(_prepared_target(), {}))
    assert family == "qwen3_8_flash_next"
    assert support.default_task == "vision_language_generation"
    assert set(support.tasks) == {"text_generation", "vision_language_generation"}


def test_partial_quantization_and_draft_are_not_the_released_target():
    config = _prepared_target()
    config["quantization_config"]["quantized_layers"].pop(
        "model.language_model.layers.47.mlp.experts"
    )
    assert describe(ModelMetadata(config, {})) is None
    config = _prepared_target()
    config["architectures"] = ["Qwen4ExpDraftForCausalLM"]
    assert describe(ModelMetadata(config, {})) is None
