# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for ministral3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    architectures=("Ministral3Model",),
    tasks=("text_to_embedding", "text_to_pooled_features", "text_to_token_features"),
    default_task="text_to_embedding",
    default_precision="bf16",
)
