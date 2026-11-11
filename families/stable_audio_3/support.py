# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Stable Audio 3.

Stable Audio 3 checkpoints are published in the native ``stable-audio-3``
layout, not as a Transformers or Diffusers repository: there is no root
``config.json`` or ``model_index.json``. The family therefore recognizes the
exact set of files that layout consists of, with the T5Gemma text encoder the
checkpoint carries in its own subfolder.
"""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata

_SENTINEL_FILES = frozenset(
    {
        "model_config.json",
        "model.safetensors",
        "t5gemma-b-b-ul2/config.json",
        "t5gemma-b-b-ul2/model.safetensors",
        "t5gemma-b-b-ul2/tokenizer.json",
    }
)
_SUPPORT = FamilySupport(tasks=("text_to_audio",), default_task="text_to_audio", default_precision="fp16")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if metadata.config or metadata.model_index:
        return None
    if _SENTINEL_FILES <= set(metadata.files):
        return _SUPPORT
    return None
