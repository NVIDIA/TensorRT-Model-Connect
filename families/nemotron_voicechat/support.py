# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Nemotron VoiceChat."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_STATIC = family_support(
    model_types=("nemotron_voicechat", "nemotronlabs_voicechat", "nvidia_nemotronlabs_voicechat_11b"),
    tasks=("speech_session",),
    default_task="speech_session",
)
_SUPPORT = FamilySupport(tasks=("speech_session",), default_task="speech_session")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if support := _STATIC(metadata):
        return support
    config = metadata.config
    model = config.get("model")
    if (
        "_rnnt_merge_info" in config
        and isinstance(model, dict)
        and "speech_generation" in model
        and "stt" in model
    ):
        return _SUPPORT
    return None
