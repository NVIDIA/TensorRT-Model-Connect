# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact sentinel identity for the published CosyVoice3 checkpoint layout."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    required = {
        "cosyvoice3.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "CosyVoice-BlankEN/config.json",
    }
    if (
        required.issubset(metadata.files)
        and metadata.model_type in ("", "cosyvoice3")
        and not metadata.pipeline_class
    ):
        return FamilySupport(
            tasks=("audio_generation",), default_task="audio_generation", default_precision="fp32"
        )
    return None
