# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for llama."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("LLaMA", "llama"),
    tasks=("text_generation", "text_prediction_positions_to_token_scores"),
    default_task="text_generation",
)
