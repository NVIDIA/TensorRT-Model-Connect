# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for vibevoice."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support

_IDENTITY = family_support(
    model_types=("vibevoice",),
    architectures=("VibeVoiceForConditionalGeneration",),
    tasks=("text_to_speech",),
    default_task="text_to_speech",
    default_precision="bf16",
)

# The original Microsoft layout names its towers like this. The Transformers
# conversion keeps the same model type but renames every sub-config and weight
# (text_config, audio_config, semantic_model_config); this family reads only the
# original layout, so the converted one is not claimed.
_ORIGINAL_LAYOUT = (
    "decoder_config",
    "acoustic_tokenizer_config",
    "semantic_tokenizer_config",
    "diffusion_head_config",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    support = _IDENTITY(metadata)
    if support is None:
        return None
    if not all(isinstance(metadata.config.get(key), dict) for key in _ORIGINAL_LAYOUT):
        return None
    return support
