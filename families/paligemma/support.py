# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact ownership of PaliGemma checkpoints with the first-generation Gemma decoder."""

from tensorrt_model_connect.model_support import family_support

_PALIGEMMA = family_support(
    model_types=("paligemma",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    default_precision="bf16",
)


def describe(metadata):
    """Claim ``paligemma`` only when its language model is Gemma; PaliGemma 2 uses Gemma 2."""
    text = metadata.config.get("text_config", {})
    # PaliGemmaConfig resolves an absent text model type to ``gemma``.
    if not isinstance(text, dict) or text.get("model_type", "gemma") != "gemma":
        return None
    return _PALIGEMMA(metadata)
