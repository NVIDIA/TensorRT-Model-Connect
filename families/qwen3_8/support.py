# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Qwen3.8."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_ALIASES = family_support(
    model_types=("qwen38", "qwen3.8", "qwen3_8"),
    tasks=("text_generation",),
    default_task="text_generation",
)
_SUPPORT = FamilySupport(tasks=("text_generation",), default_task="text_generation")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if support := _ALIASES(metadata):
        return support
    config = metadata.config.get("text_config", metadata.config)
    if not isinstance(config, dict):
        return None
    model_type = str(metadata.config.get("model_type", "")).lower().replace(".", "_")
    if model_type not in {"qwen3_5", "qwen3_8"}:
        return None
    if "mlp_only_layers" in config or "output_gate_type" not in config:
        return None
    return _SUPPORT
