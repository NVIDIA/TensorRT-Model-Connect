# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support

ARCHITECTURE = "MossTranscribeDiarizeForConditionalGeneration"

_describe_identity = family_support(
    model_types=("moss_transcribe_diarize",),
    architectures=(ARCHITECTURE,),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    # Other heads of this model type (the bare backbone produces no transcript)
    # are refused by the build, so a checkpoint that names its architectures is
    # claimed only when one of them is the conditional-generation head.
    if metadata.architectures and ARCHITECTURE not in metadata.architectures:
        return None
    return _describe_identity(metadata)
