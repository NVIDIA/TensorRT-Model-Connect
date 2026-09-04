# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for qwen3_5."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_BASE = family_support(
    model_types=("qwen35", "qwen3.5", "qwen3_5"),
    tasks=("text_generation",),
    default_task="text_generation",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    config = metadata.config.get("text_config", metadata.config)
    if isinstance(config, dict) and "output_gate_type" in config and (
        "mlp_only_layers" not in config
    ):
        return None
    return _BASE(metadata)
