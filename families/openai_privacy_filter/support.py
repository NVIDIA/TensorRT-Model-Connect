# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for openai_privacy_filter."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("openai_privacy_filter",),
    architectures=("OpenAIPrivacyFilterForTokenClassification",),
    tasks=("text_to_head_scores",),
    default_task="text_to_head_scores",
)
