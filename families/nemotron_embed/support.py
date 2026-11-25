# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    text = metadata.config.get("text_config", {})
    if (
        metadata.model_type == "mistral3"
        and text.get("model_type") == "ministral3"
        and text.get("is_causal") is False
        and metadata.config.get("pooling") == "avg"
    ):
        return FamilySupport(
            tasks=("text_to_embedding",), default_task="text_to_embedding", default_precision="fp32"
        )
    return None
