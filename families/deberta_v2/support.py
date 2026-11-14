# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for deberta_v2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("deberta-v2", "deberta_v2"),
    architectures=("DebertaV2Model",),
    tasks=("text_to_token_features",),
    default_task="text_to_token_features",
)
