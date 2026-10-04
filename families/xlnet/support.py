# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for xlnet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("xlnet",),
    tasks=("text_to_embedding", "text_to_pooled_features", "text_pair_to_relevance"),
    default_task="text_to_pooled_features",
)
