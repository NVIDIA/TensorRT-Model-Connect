# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support

MODEL_TYPE = "moss_transcribe_diarize"
ARCHITECTURE = "MossTranscribeDiarizeForConditionalGeneration"

_describe_identity = family_support(
    model_types=(MODEL_TYPE,),
    architectures=(ARCHITECTURE,),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    # The build reads config.json's model_type and its plural "architectures"
    # list, nothing else. Discovery refuses what either of them contradicts (the
    # bare backbone produces no transcript), and still claims a checkpoint that
    # names only one of them.
    model_type = metadata.config.get("model_type")
    if isinstance(model_type, str) and model_type and model_type != MODEL_TYPE:
        return None
    listed = metadata.config.get("architectures")
    if listed is not None:
        names = (listed,) if isinstance(listed, str) else listed if isinstance(listed, list) else ()
        if ARCHITECTURE not in names:
            return None
    return _describe_identity(metadata)
