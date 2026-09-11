# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for gpt2."""

# Comment-only live-fire payload for the ordered Community CI introduced in
# PR #1262. This does not change GPT-2 support or runtime behavior.

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("gpt2",),
    tasks=("text_generation",),
    default_task="text_generation",
)
