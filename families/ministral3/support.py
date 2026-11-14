# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for ministral3."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_ENCODER = family_support(
    architectures=("Ministral3Model",),
    tasks=("text_to_embedding", "text_to_pooled_features", "text_to_token_features"),
    default_task="text_to_embedding",
    default_precision="bf16",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    """Only bidirectional encoders: a ``Ministral3Model`` without ``is_causal: false`` is causal."""
    if metadata.config.get("is_causal") is not False:
        return None
    return _ENCODER(metadata)
