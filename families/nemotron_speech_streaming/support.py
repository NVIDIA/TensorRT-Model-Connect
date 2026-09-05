# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for nemotron_speech_streaming."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("enc_dec_rnnt_bpe", "fastconformer_cacheaware_rnnt", "nemotron3_5_asr", "nemotron_asr_streaming", "nemotron_speech_streaming", "nemotronspeechstreaming", "rnnt_bpe"),
    architectures=("Nemotron3_5AsrForRNNT",),
    tasks=("transcription_streaming",),
    default_task="transcription_streaming",
)
