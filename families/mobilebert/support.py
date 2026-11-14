# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for MobileBERT."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("mobilebert",),
    tasks=("masked_text_to_token_scores", "text_pair_to_pretraining_relation_scores"),
    default_task="masked_text_to_token_scores",
)
