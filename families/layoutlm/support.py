# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for LayoutLM document question answering.

Only the extractive question-answering head is owned. The other LayoutLM heads
(masked LM, sequence and token classification, the bare encoder) resolve to no
family, which reports them as unsupported instead of building the wrong head.
"""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    architectures=("LayoutLMForQuestionAnswering",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
)
