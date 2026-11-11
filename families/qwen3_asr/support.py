# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint identity and task support for Qwen3-ASR."""

from tensorrt_model_connect.model_support import family_support

_ARCHITECTURE = "Qwen3ASRForConditionalGeneration"

_transcription = family_support(
    model_types=("qwen3_asr",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)


def describe(metadata):
    """Claim the Transformers-native speech-recognition checkpoints only.

    * The forced-aligner head (``Qwen3ASRForTokenClassification``) shares ``model_type``
      ``qwen3_asr`` but predicts timestamps, which this family does not build.
    * The original Qwen3-ASR releases (``Qwen/Qwen3-ASR-0.6B`` and siblings without ``-hf``)
      declare the same ``model_type`` and architecture but nest their configuration under
      ``thinker_config``; this family reads the Transformers layout, with top-level
      ``audio_config`` and ``text_config``.
    """
    architectures = metadata.architectures
    if architectures and architectures != (_ARCHITECTURE,):
        return None
    config = metadata.config
    if "thinker_config" in config:
        return None
    if not isinstance(config.get("audio_config"), dict) or not isinstance(
        config.get("text_config"), dict
    ):
        return None
    return _transcription(metadata)
