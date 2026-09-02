# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-run timing contracts shared by the release orchestrator and baseline runners."""

from __future__ import annotations

from typing import Any


MODEL_CALL_FAMILIES = {
    "bark",
    "chronos_bolt",
    "dinov3",
    "eagle_vlm",
    "elf_flow",
    "internvl",
    "locateanything",
    "parakeet_tdt",
    "patchtsmixer",
    "patchtst",
    "phi4_multimodal",
    "qwen3_omni",
    "qwen_vl",
    "sam",
    "sam3",
    "segformer",
    "timesfm",
    "timm_vit",
    "whisper",
}

ASSET_IN_TIMED_CALL_FAMILIES = {
    "canary",
    "deepseek_ocr",
    "lance",
    "nemotron_speech_streaming",
}


def timing_contract(
    *,
    runner: str,
    family: str,
) -> dict[str, Any]:
    """Return the declared reference boundary and its matching TRTMC boundary."""
    if runner == "hf-transformers":
        return {
            "timing_scope": "public_operation_call_wall",
            "candidate_timing_scope": "public_pipeline_call_wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        }
    if runner != "task-reference":
        raise ValueError(f"unsupported baseline runner: {runner}")
    if family in MODEL_CALL_FAMILIES:
        return {
            "timing_scope": "task-model-call-wall",
            "candidate_timing_scope": "model_call_wall",
            "input_preparation_included": False,
            "asset_loading_included": False,
        }
    return {
        "timing_scope": "task-pipeline-call-wall",
        "candidate_timing_scope": "public_pipeline_call_wall",
        "input_preparation_included": True,
        "asset_loading_included": family in ASSET_IN_TIMED_CALL_FAMILIES,
    }
