# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify the prepared multimodal Flash-Next NVFP4 target, not a draft."""

from tensorrt_model_connect.model_support import FamilySupport, Variant

TASKS = ("text_generation", "vision_language_generation")


def is_flash_next(cfg):
    text = cfg.get("text_config") or {}
    vision = cfg.get("vision_config") or {}
    quant = cfg.get("quantization_config") or {}
    layers = quant.get("quantized_layers") or {}
    return (
        cfg.get("model_type") == "qwen4_exp"
        and cfg.get("architectures") == ["Qwen4ExpForConditionalGeneration"]
        and text.get("model_type") == "qwen4_exp_text"
        and text.get("num_hidden_layers") == 48
        and text.get("num_experts") == 512
        and text.get("num_experts_per_tok") == 10
        and text.get("ple_layer_ids") == [2]
        and vision.get("out_hidden_size") == text.get("hidden_size") == 2560
        and quant.get("quant_method") == "modelopt"
        and all(
            layers.get(f"model.language_model.layers.{i}.mlp.experts")
            == {"quant_algo": "NVFP4", "group_size": 16}
            for i in range(48)
        )
    )


def describe(metadata):
    if not is_flash_next(metadata.config):
        return None
    return FamilySupport(
        tasks=TASKS,
        default_task=TASKS[1],
        default_precision="fp16",
        variants=(Variant("fp16", status="experimental"),),
    )
