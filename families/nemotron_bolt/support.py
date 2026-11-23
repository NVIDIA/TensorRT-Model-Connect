# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only claim the prepared multimodal Puzzle identity, never its qwen3 draft on its own."""

from tensorrt_model_connect.model_support import FamilySupport, Variant

TASKS = ("text_generation", "vision_language_generation")


def is_bolt(cfg: dict) -> bool:
    text = cfg.get("llm_config")
    vision = cfg.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        return False
    blocks = text.get("block_configs")
    return (
        cfg.get("model_type") == "nemotron_h_omni"
        and cfg.get("architectures") == ["NemotronH_Omni_Reasoning_V3"]
        and text.get("model_type") == "nemotron_h_puzzle"
        and text.get("architectures") == ["NemotronHPuzzleForCausalLM"]
        and text.get("hidden_size") == 4096
        and text.get("vocab_size") == 131072
        and isinstance(blocks, list)
        and len(blocks) == 88
        and vision.get("model_type") == "radio"
        and cfg.get("sound_config") is None
    )


def describe(metadata):
    if not is_bolt(metadata.config):
        return None
    return FamilySupport(
        tasks=TASKS,
        default_task="vision_language_generation",
        default_precision="bf16",
        variants=(Variant("bf16", status="experimental"),),
    )
