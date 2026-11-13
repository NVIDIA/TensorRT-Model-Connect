# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for ESM-2."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("esm",),
    architectures=("EsmForMaskedLM", "EsmModel"),
    tasks=("masked_text_to_token_scores",),
    default_task="masked_text_to_token_scores",
)
